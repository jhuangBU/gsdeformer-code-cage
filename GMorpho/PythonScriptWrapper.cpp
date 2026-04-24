#include "PythonScriptWrapper.h"

#include <QProcess>
#include <QTemporaryDir>
#include <QCryptographicHash>
#include <cassert>

#include "npy.h"

static std::tuple<int, std::string, std::string> invoke_process(const std::string& program, const std::vector<std::string>& arguments,
                                                                const QProcessEnvironment& env) {
    QProcess proc;
    // TODO: make it an option and tells user stdout/stderr will be empty if enabled!
    proc.setProcessChannelMode(QProcess::ProcessChannelMode::ForwardedChannels);
    proc.setProcessEnvironment(env);

    QStringList args;
    QString debugArgs = QString::fromStdString(program);
    for (const auto &item: arguments) {
        auto i = QString::fromStdString(item);
        args.append(i);
        debugArgs.append(" " + i);
    }

    // std::cout << "Invoking: " << debugArgs.toStdString() << std::endl;

    proc.start(program.c_str(), args);

    if (!proc.waitForStarted())
        throw std::runtime_error("error starting process: " + debugArgs.toStdString());

    if (!proc.waitForFinished(-1))
        throw std::runtime_error("error running process: " + debugArgs.toStdString());

    auto sout = proc.readAllStandardOutput().toStdString();
    auto serr = proc.readAllStandardError().toStdString();
    return std::tuple<int, std::string, std::string>{proc.exitCode(), sout, serr};
}

void invoke_python_script(const std::string& path_to_script, const std::vector<std::string>& arguments,
                          const std::string& conda_env) {
    std::string prog = "conda";
    std::vector<std::string> args {"run", "-n", conda_env, "--no-capture-output", "python"};
    args.emplace_back(path_to_script);
    args.insert(args.end(), arguments.begin(), arguments.end());

    // conda run does not run conda activation scripts, so LD_LIBRARY_PATH is not
    // set up automatically. Prepend the conda env's lib dir so native libs
    // (e.g. libboost_thread.so) are found by the dynamic linker.
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QString condaRoot = env.value("CONDA_ROOT",
                                  env.value("MAMBA_ROOT_PREFIX",
                                            env.value("CONDA_PREFIX", "")));
    if (!condaRoot.isEmpty()) {
        QString envLib = condaRoot + "/envs/" + QString::fromStdString(conda_env) + "/lib";
        QString ldPath = env.value("LD_LIBRARY_PATH", "");
        env.insert("LD_LIBRARY_PATH", ldPath.isEmpty() ? envLib : envLib + ":" + ldPath);
    }

    auto ret = invoke_process(prog, args, env);
    if (std::get<0>(ret) != 0) {
        auto msg =
                std::string {"error running python script: "} + path_to_script
                + "\nstdout: " + std::get<1>(ret)
                + "\nstderr: " + std::get<2>(ret)
        ;
        throw std::runtime_error(msg);
    }
}

std::vector<MorphoGraphics::Vec3f> extractVector3f(const npy::npy_data<float>& pret) {
    // read result: float array of shape (N,3)
    assert(pret.shape.size() == 2);
    assert(pret.shape[1] == 3);
    auto pret_data = pret.data;

    // fetching results into it
    std::vector<MorphoGraphics::Vec3f> ret;
    auto count = pret_data.size() / 3;
    ret.reserve(count);
    for(decltype(count) i=0;i<count;i+=1) {
        ret.emplace_back(pret_data[i*3], pret_data[i*3+1], pret_data[i*3+2]);
    }
    return ret;
}

std::vector<float> extractVector1f(const npy::npy_data<float>& pret) {
    // read result: float array of shape (1)
    assert(pret.shape.size() == 1);
    auto pret_data = pret.data;
    return pret_data;
}

npy::npy_data<float> encodeVector3f(const std::vector<MorphoGraphics::Vec3f>& data) {
    npy::npy_data<float> d;
    d.shape = {data.size(), 3};
    d.data.resize(data.size()*3);
    for (size_t i = 0; i < data.size(); ++i) {
        auto& v = data[i];
        d.data[i*3] = v[0];
        d.data[i*3+1] = v[1];
        d.data[i*3+2] = v[2];
    }
    d.fortran_order = false; // optional
    return d;
}

std::vector<MorphoGraphics::Vec3f> extract_means(const std::string& path_3dgs_folder) {
    using MorphoGraphics::Vec3f;

    QTemporaryDir dir;
    if (!dir.isValid()) {
        throw std::runtime_error(std::string{"cannot create tempdir "} + dir.errorString().toStdString());
    }
    auto output_file = dir.path().toStdString() + "/" + "tmp_mean.npy";

    invoke_python_script("python_scripts/extract_means.py", {
            "--model_folder", path_3dgs_folder,
            "--output_file", output_file
    });

    auto pret = npy::read_npy<float>(output_file);
    std::vector<Vec3f> ret = extractVector3f(pret);

    return ret;
}

/**
 * locating point_cloud.ply file in folder, with max iter when iteration==-1
 */
QString _locate_ply_file(const std::string& path_3dgs_folder, int iteration=-1) {
    QDir dir(QString::fromStdString(path_3dgs_folder));

    // assert dirs exist
    assert(dir.exists());
    assert(dir.cd("point_cloud"));

    // decide iterations
    if(iteration == -1) {
        auto list = dir.entryList(QStringList() << "iteration_*");
        int max = -1;
        for (const auto &item: list) {
            bool ok = false;
            int it = item.split("_")[1].toInt(&ok);
            assert(ok);
            max = max > it ? max : it;
        }
        assert(max != -1);
        iteration = max;
    }
    dir.cd("iteration_" + QString::number(iteration));

    // go to point_cloud.ply
    auto ret = dir.filePath("point_cloud.ply");

    return ret;
}

std::string locate_3dgs_ply(const std::string & path_3dgs_folder, int iteration) {
    return _locate_ply_file(path_3dgs_folder, iteration).toStdString();
}

QString _hash_file(const QString& file) {
    QFile f(file);
    assert(f.open(QIODevice::ReadOnly));
    QCryptographicHash hash(QCryptographicHash::Algorithm::Sha256);
    hash.addData(&f);
    return hash.result().toHex();
}

const QString DENSITY_CACHE_HASH_FILE = "density_cache_key.hash";
const QString DENSITY_CACHE_FILE = "density_cache.npy";

std::unique_ptr<std::vector<float>> cache_check_compute_densities(const std::string & path_3dgs_folder) {
    auto ply_file = _locate_ply_file(path_3dgs_folder);
    auto ply_hash = _hash_file(ply_file);

    bool use_cache = QFile(DENSITY_CACHE_HASH_FILE).exists(); // hash file exists
    QFile hash_f(DENSITY_CACHE_HASH_FILE);
    hash_f.open(QIODevice::ReadOnly);
    use_cache = use_cache && QString::fromUtf8(hash_f.readAll()) == ply_hash; // hash is same
    use_cache = use_cache && QFile(DENSITY_CACHE_FILE).exists(); // cache exists
    std::cout << "debug - cache - " << use_cache << " - ply " << ply_file.toStdString() << " - hash " << ply_hash.toStdString() << std::endl;
    if (use_cache) {
        auto read_npy = npy::read_npy<float>(DENSITY_CACHE_FILE.toStdString());
        std::unique_ptr<std::vector<float>> ret{new std::vector<float>()};
        *ret = extractVector1f(read_npy);
        return ret;
    } else {
        return nullptr;
    }
}

void cache_write_compute_densities(const std::vector<float>& densities, const std::string & path_3dgs_folder) {
    auto ply_file = _locate_ply_file(path_3dgs_folder);
    auto ply_hash = _hash_file(ply_file);

    QFile hashf(DENSITY_CACHE_HASH_FILE);
    hashf.open(QIODevice::WriteOnly);
    hashf.write(ply_hash.toUtf8());

    npy::npy_data_ptr<float> data_ptr;
    data_ptr.data_ptr = densities.data();
    data_ptr.shape = {densities.size()};
    data_ptr.fortran_order = false;
    npy::write_npy(DENSITY_CACHE_FILE.toStdString(), data_ptr);
}

std::vector<float> compute_densities(
        std::vector<MorphoGraphics::Vec3f>&& coordinates,
        const std::string & path_3dgs_folder
) {
    using MorphoGraphics::Vec3f;

    QTemporaryDir dir;
    if (!dir.isValid()) {
        throw std::runtime_error(std::string{"cannot create tempdir "} + dir.errorString().toStdString());
    }
    auto coord_file = dir.path().toStdString() + "/" + "tmp_coord.npy";
    auto density_file = dir.path().toStdString() + "/" + "tmp_density.npy";

    auto cnpy = encodeVector3f(coordinates);
    npy::write_npy(coord_file, cnpy);
    coordinates.clear();
    coordinates.shrink_to_fit();
    cnpy.data.clear();
    cnpy.data.shrink_to_fit();

    invoke_python_script("python_scripts/compute_densities.py", {
            "--model_folder", path_3dgs_folder,
            "--coord_file", coord_file,
            "--output_file", density_file
    });

    auto pret = npy::read_npy<float>(density_file);
    return extractVector1f(pret);
}

std::vector<unsigned char> compute_packed_voxel_grid_from_tsdf(
    const std::string & path_3dgs_folder,
    MorphoGraphics::AxisAlignedBoundingBox bbox, int res, int num_bit_vox
) {
    using MorphoGraphics::Vec3f;

    QTemporaryDir dir;
    if (!dir.isValid()) {
        throw std::runtime_error(std::string{"cannot create tempdir "} + dir.errorString().toStdString());
    }
    auto ret_file = dir.path().toStdString() + "/" + "test_packed_voxels.npy";

    char bbox_min_str[64];
    char bbox_max_str[64];
    sprintf(bbox_min_str, "%.6f,%.6f,%.6f", bbox.min()[0], bbox.min()[1], bbox.min()[2]);
    sprintf(bbox_max_str, "%.6f,%.6f,%.6f", bbox.max()[0], bbox.max()[1], bbox.max()[2]);

    invoke_python_script("python_scripts/tsdf_voxelize.py", {
            // model options
            "--model_path", path_3dgs_folder,
            // camera options
            "--fx", "800",
            "--fy", "800",
            "--width", "800",
            "--height", "800",
            // T-SDF options
            "--tsdf_resolution", std::to_string(res),
            // voxel options
            "--resolution", std::to_string(res),
            "--bbox_max="+std::string{bbox_max_str}, 
            "--bbox_min="+std::string{bbox_min_str},
            // output options
            "--expname", "test",
            "--output_path", dir.path().toStdString() 
    });

    auto pret = npy::read_npy<unsigned char>(ret_file);
    // read result: unsigned char array of shape (1)
    assert(pret.shape.size() == 1);
    return pret.data;
}