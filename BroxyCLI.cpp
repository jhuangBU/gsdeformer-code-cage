#include <iostream>

#include <QApplication>
#include <QSurfaceFormat>
#include <QWindow>
#include <utility>
#include <QJsonDocument>
#include <QJsonObject>
#include <QFile>

// #include "GSVoxelizer.h"
#include "BroxyCLI.h"
#include "Voxelizer.h"
#include "GMorpho.h"
#include "FrameField.h"
#include "Decimator.h"

using namespace MorphoGraphics;

static const float MAX_SE_SIZE = 0.3f;
static const float INITIAL_SE_SIZE = 0.01f;
static const unsigned int VOXEL_RESOLUTION = 256;

struct CLIStates {
    Mesh mesh_morpho_buf;
    GMorpho gmorpho;
    ScaleField scale_field;
    FrameField frame_field;
};

void load_mesh(
    CLIStates& states,
    const DummyViewerConfig& config,
    bool bilateralMorphoMeshNormals = true
) {
    try {
        // preconfigure
        states.gmorpho.set_bilateral_filtering (bilateralMorphoMeshNormals);

        if (config.file_type == "mesh") {
            Mesh mesh_input_buf;

            // read mesh file
            mesh_input_buf.load (config.file_path);

            // load mesh into gmorpho
            const std::vector<Vec3f> & positions = mesh_input_buf.P ();
            const std::vector<Vec3f> & normals = mesh_input_buf.N ();
            const std::vector< Vec3<unsigned int> > & triangles = mesh_input_buf.T ();
            states.gmorpho.Load (&positions[0][0], &normals[0][0], positions.size (),
                                 &triangles[0][0], triangles.size (), VOXEL_RESOLUTION, // VOXEL RESOLUTION
                                 MAX_SE_SIZE);
        } else if (config.file_type == "3dgs") {
            if(config.gs_threshold != 0.0) {
                states.gmorpho.Load3DGS(config.file_path, VOXEL_RESOLUTION, MAX_SE_SIZE, config.gs_threshold, !config.benchmark);
            } else {
                states.gmorpho.Load3DGS(config.file_path, VOXEL_RESOLUTION, MAX_SE_SIZE, 1e-6, !config.benchmark);
            }
        } else {
            throw std::runtime_error("unrecognized filetype: " + config.file_type);
        }

        // init scale field
        Vec3f bbox_min = states.gmorpho.bbox ().min();
        Vec3f bbox_max = states.gmorpho.bbox ().max();
        Vec3ui res = ((unsigned int)2) * states.gmorpho.res ();
        float cell_size = 0.5f * states.gmorpho.cell_size ();
        states.scale_field.set_global_scale (config.global_scale != 0.0 ? config.global_scale : INITIAL_SE_SIZE);
        states.scale_field.Init (bbox_min, bbox_max, res, cell_size);
        states.scale_field.UpdateGridGlobalScale ();

        // reset brushing configs
        states.gmorpho.set_use_asymmetric_closing (true);
        states.gmorpho.set_use_frame_field (false);
        //	gmorpho.set_use_asymmetric_closing (false);

        // init frame field
        bool update_only = false;
        bool use_group_symmetry = false;
        Vec3i frame_field_res (states.gmorpho.res ()[0] / 16,
                               states.gmorpho.res ()[1] / 16,
                               states.gmorpho.res ()[2] / 16);
        std::cout << "frame_field_res : " << frame_field_res << std::endl;
        states.frame_field.Init (bbox_min, bbox_max, frame_field_res,
                          FrameField::BiHarmonicSystem
                //										FrameField::LocalOptimization
        );
        states.frame_field.Generate (update_only, use_group_symmetry);

        // Compute the morphological transformation from
        // the Scale Field and the Frame Field
        states.gmorpho.Update (states.mesh_morpho_buf, states.scale_field, states.frame_field);
    } catch (Mesh::Exception & e) {
        std::cerr << e.msg () << std::endl;
        exit (1);
    }
}

void decimate_to_cage(CLIStates& states, DummyViewerConfig config, Mesh& cage) {
    std::vector<MorphoGraphics::Vec3f> _cageTriNormals;

    std::vector<bool> feature_taggs (states.mesh_morpho_buf.P ().size (), false);
    ScaleField & scale_field = states.scale_field;
    char * scale_grid_cpu = NULL;
    Vec3f bbox_min = scale_field.bbox_min ();
    Vec3f bbox_max = scale_field.bbox_max ();
    float cell_size = scale_field.cell_size ();
    Vec3<unsigned int> res = scale_field.res ();
    scale_field.GetScaleGridCPU (&scale_grid_cpu);
    bool use_linear_constraints = true, use_features = false;
    Decimator * decimator = NULL;
    double time1, time2;
    float target_error;
    int target_num_faces;
    float max_edge_length_alpha;

    if (states.gmorpho.use_asymmetric_closing ()) {
        target_error = 0.01f;
        target_num_faces = 20;
        max_edge_length_alpha = 2.f;
    } else {
        target_error = 0.001f;
        target_num_faces = 20;
        max_edge_length_alpha = 1.f;
    }

    auto d = config.params;
    target_error = d.target_error;
    target_num_faces = d.target_num_faces;
    auto target_num_vertices = d.target_num_vertices;
    max_edge_length_alpha = d.max_edge_length_alpha;

    // Print decimation parameters in JSON format
    std::cout << "performing decimation with parameter: {\n"
              << "  \"target_error\": " << d.target_error << ",\n"
              << "  \"target_num_faces\": " << d.target_num_faces << ",\n"
              << "  \"target_num_vertices\": " << d.target_num_vertices << ",\n"
              << "  \"max_edge_length_alpha\": " << d.max_edge_length_alpha << "\n"
              << "}" << std::endl;

    time1 = GET_TIME ();
    decimator = new Decimator (states.mesh_morpho_buf, target_num_faces,
                               target_error,
                               use_linear_constraints,
                               use_features,
                               max_edge_length_alpha,
                               bbox_min, bbox_max,
                               res, cell_size,
                               scale_grid_cpu,
                               feature_taggs,
                               3 // HC iterations
    );

    bool heap_not_empty = true;
    while (true) {
        bool has_more_faces = decimator->num_faces() > decimator->target_num_faces();
        bool has_more_vertices = decimator->num_vertices() > target_num_vertices;
        bool below_target_error = decimator->error() < decimator->target_error();
        if(!(heap_not_empty && (has_more_vertices && has_more_faces) && below_target_error)) {
            break;
        }
        heap_not_empty = decimator->Optimize ();
        decimator->GetMesh (cage);
        cage.recomputeNormals();
        cage.computeTriNormals (_cageTriNormals);
    }
    time2 = GET_TIME ();

    decimator->GetMesh (cage);

    delete scale_grid_cpu;
    std::cout << "Feature-Aware Decimator error : "
              << decimator->error ()
              << " and number of faces : "
              << cage.T ().size ()
              << " in " << time2 - time1 << " ms."
              << std::endl;
}

void set_use_asymmetric_closing (CLIStates& states, bool b) {
    states.gmorpho.set_use_asymmetric_closing (b);
    double time1 = GET_TIME ();
    states.gmorpho.Update (states.mesh_morpho_buf, states.scale_field, states.frame_field);
    double time2 = GET_TIME ();
    std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
}

void set_use_rotation_field (CLIStates& states, bool b) {
    states.gmorpho.set_use_frame_field (b);
    double time1 = GET_TIME ();
    states.gmorpho.Update (states.mesh_morpho_buf, states.scale_field, states.frame_field);
    double time2 = GET_TIME ();
    std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
}

void increase_base_scale (CLIStates& states) {
    double time1, time2;
    // calculate new global_scale
    float global_scale = states.scale_field.global_scale ();
    global_scale += 0.5f * states.gmorpho.cell_size ();

    // bounding, only eval when in bound
    global_scale = std::max (global_scale,
                             0.5f * states.gmorpho.cell_size ());
    global_scale = std::min (global_scale,
                             MAX_SE_SIZE);
    if ((global_scale != MAX_SE_SIZE) && (global_scale != 0.5f * states.gmorpho.cell_size ())) {
        // set, time, update, Update(), updateGL()
        states.scale_field.set_global_scale (global_scale);
        time1 = GET_TIME ();
        states.scale_field.UpdateGridGlobalScale ();
        states.gmorpho.Update (states.mesh_morpho_buf, states.scale_field, states.frame_field);
        time2 = GET_TIME ();
        std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
    }
}

void decrease_base_scale (CLIStates& states) {
    double time1, time2;
    // calculate new global_scale
    float global_scale = states.scale_field.global_scale ();
    global_scale -= 0.5f * states.gmorpho.cell_size ();

    // bounding, only eval when in bound
    global_scale = std::max (global_scale,
                             0.5f * states.gmorpho.cell_size ());
    global_scale = std::min (global_scale,
                             MAX_SE_SIZE);
    if ((global_scale != MAX_SE_SIZE) && (global_scale != 0.5f * states.gmorpho.cell_size ())) {
        // set, time, update, Update(), updateGL()
        states.scale_field.set_global_scale (global_scale);
        time1 = GET_TIME ();
        states.scale_field.UpdateGridGlobalScale ();
        states.gmorpho.Update (states.mesh_morpho_buf, states.scale_field, states.frame_field);
        time2 = GET_TIME ();
        std::cout << "[Morpho Update] : " << time2 - time1 << " ms." << std::endl;
    }

}

DummyViewer::DummyViewer (QWidget * parent, DummyViewerConfig config)
: QGLViewer (parent), _config(std::move(config)) {}

DummyViewer::~DummyViewer () {}

void DummyViewer::init () {
    // init context
    std::cout << "[Status] Configuring GLEW" << std::endl;
    glewInit();
    if (!GLEW_VERSION_3_0) {
        std::cerr << "Driver does not support OpenGL v3.0" << std::endl;
        exit (EXIT_FAILURE);
    }
//    setAutoBufferSwap (true);
//    setBackgroundColor (QColor (255, 255, 255));
//    glEnable (GL_BLEND);
//    glLineWidth (2.f);
//    glEnable (GL_LINE_SMOOTH);
//    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // run algorithm
    std::cout << "[Status] Running algorithm" << std::endl;

    CLIStates states;
    Mesh cage;
    auto count = _config.benchmark ? 10 : 1;
    for (int i = 0; i < count; ++i) {
        auto time1 = GET_TIME ();
        load_mesh(states, _config);
        decimate_to_cage(states, _config, cage);
        cage.store(_config.output_path);
        auto time2 = GET_TIME ();
        if(_config.benchmark) {
            std::cout << "--- Benchmark - Run" << i << " - " << time2 - time1 << " ms ---" << std::endl;
        }
    }

    // close the window for exit
    QMetaObject::invokeMethod(this, "close", Qt::QueuedConnection);
}

DummyViewerConfig parse_config_json(const std::string& path) {
    DummyViewerConfig config;

    // Open and read the JSON file
    QFile file(QString::fromStdString(path));
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("failed to open file: " + path);
    }

    // Parse JSON
    QJsonDocument jsonDoc = QJsonDocument::fromJson(file.readAll());
    file.close();

    if (jsonDoc.isNull()) {
        throw std::runtime_error("failed to parse json");
    }

    QJsonObject jsonObj = jsonDoc.object();

    // Parse DummyViewerConfig fields
    config.benchmark = jsonObj["benchmark"].toBool(false);
    config.global_scale = jsonObj["global_scale"].toDouble(0.0);
    config.file_path = jsonObj["file_path"].toString().toStdString();
    config.file_type = jsonObj["file_type"].toString().toStdString();
    config.gs_threshold = jsonObj["gs_threshold"].toDouble(0.0);
    config.output_path = jsonObj["output_path"].toString().toStdString();

    // Parse DecimationParams
    if (jsonObj.contains("params") && jsonObj["params"].isObject()) {
        QJsonObject paramsObj = jsonObj["params"].toObject();
        config.params.target_error = paramsObj["target_error"].toDouble(5.0f);
        config.params.target_num_faces = paramsObj["target_num_faces"].toInt(20);
        config.params.target_num_vertices = paramsObj["target_num_vertices"].toInt(150);
        config.params.max_edge_length_alpha = paramsObj["max_edge_length_alpha"].toDouble(1.0f);
    }

    return config;
}

std::string to_json(const DummyViewerConfig& config)
{
    QJsonObject paramsJson;
    paramsJson["target_error"] = config.params.target_error;
    paramsJson["target_num_faces"] = config.params.target_num_faces;
    paramsJson["target_num_vertices"] = config.params.target_num_vertices;
    paramsJson["max_edge_length_alpha"] = config.params.max_edge_length_alpha;

    QJsonObject configJson;
    configJson["benchmark"] = config.benchmark;
    configJson["global_scale"] = config.global_scale;
    configJson["params"] = paramsJson;
    configJson["file_path"] = QString::fromStdString(config.file_path);
    configJson["file_type"] = QString::fromStdString(config.file_type);
    configJson["gs_threshold"] = config.gs_threshold;
    configJson["output_path"] = QString::fromStdString(config.output_path);

    QJsonDocument doc(configJson);

    QString jsonString = doc.toJson(QJsonDocument::Indented);

    return jsonString.toStdString();
}

int main_cli(int argc, char ** argv) {
    QApplication app(argc, argv);
    DummyViewerConfig config;
    if(argc>=3) {
        config = parse_config_json(std::string(argv[2]));
    }
    std::cout << "Running with Config:" << std::endl;
    std::cout << to_json(config) << std::endl;
    DummyViewer viewer(nullptr, config);
    viewer.show();
    auto ret = app.exec();
    std::cout << "[Status] Execution Completed." << std::endl;
    return ret;
}


int main_gs_cli(int argc, char ** argv) {
//    GSVoxelizer voxelizer;
//    voxelizer.Load("TestSamples/nerf_lego");
//    auto base_res = VOXEL_RESOLUTION;
//    auto se_size_ = MAX_SE_SIZE;
//    float margin = se_size_ + 2.f / (2 * ((float)base_res));
//    voxelizer.VoxelizeConservative(base_res, margin);
    return 0;
}