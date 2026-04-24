//
// Created by johnbanq on 5/5/24.
//

#ifndef BROXY_BROXYCLI_H
#define BROXY_BROXYCLI_H

#define GLEW_STATIC 1
#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <BroxyViewer.h>

struct DummyViewerConfig {
    bool benchmark = false;
    float global_scale = 0.0;
    DecimationParams params = DecimationParams();
    std::string file_path = "Resources/Models/beast.off";
    std::string file_type = "mesh";
    float gs_threshold = 0.0;
    std::string output_path = "beast_cage_cli.off";
};

/*
 * dummy viewer for creating OpenGL context for operation
*/
class DummyViewer : public QGLViewer  {
Q_OBJECT
public:
    DummyViewer (QWidget * parent = 0, DummyViewerConfig config = DummyViewerConfig());
    virtual ~DummyViewer ();
protected:
    void init();
private:
    DummyViewerConfig _config;
};

int main_cli(int argc, char ** argv);

int main_gs_cli(int argc, char ** argv);

#endif //BROXY_BROXYCLI_H
