/*
	Copyright (c) 2015-2017 Telecom ParisTech (France).
	Authors: Stephane Calderon and Tamy Boubekeur.
	All rights reserved.

	This file is part of Broxy, the reference implementation for
	the paper:
		Bounding Proxies for Shape Approximation.
		Stephane Calderon and Tamy Boubekeur.
		ACM Transactions on Graphics (Proc. SIGGRAPH 2017),
		vol. 36, no. 5, art. 57, 2017.

	You can redistribute it and/or modify it under the terms of the GNU
	General Public License as published by the Free Software Foundation,
	either version 3 of the License, or (at your option) any later version.

	Licensees holding a valid commercial license may use this file in
	accordance with the commercial license agreement provided with the software.

	This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
	WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "BroxyApp.h"

#include <QAction>
#include <QActionGroup>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QFileDialog>
#include <QGroupBox>
#include <QCheckBox>
#include <QMenu>
#include <QMenuBar>
#include <QString>
#include <QDebug>
#include <QInputDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

using namespace std;

void BroxyApp::saveCamera() {
	qglviewer::Vec position = m_viewer->camera()->position();
	qglviewer::Quaternion orientation = m_viewer->camera()->orientation();
    
    QJsonObject cameraJson;
    
    // Save position as array
    QJsonArray positionArray;
    positionArray.append(position.x);
    positionArray.append(position.y);
    positionArray.append(position.z);
    cameraJson["position"] = positionArray;
    
    // Save orientation as array
    QJsonArray orientationArray;
    orientationArray.append(orientation[0]);
    orientationArray.append(orientation[1]); 
    orientationArray.append(orientation[2]);
    orientationArray.append(orientation[3]);
    cameraJson["orientation"] = orientationArray;
    
    QJsonDocument doc(cameraJson);
    QString filename = QFileDialog::getSaveFileName(m_mainWindow,  "Save Camera",  QDir::currentPath() + "/saved_camera.json", tr("JSON files (*.json)"));

    if (!filename.isNull()) {
        QFile file(filename);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(doc.toJson());
            file.close();
        }
    }
}

void BroxyApp::loadCamera() {
    QString filename = QFileDialog::getOpenFileName(m_mainWindow, "Load Camera", QDir::currentPath(), tr("JSON files (*.json)"));
    if (!filename.isNull()) {
        QFile file(filename);
        if (file.open(QIODevice::ReadOnly)) {
            QByteArray data = file.readAll();
            QJsonDocument doc = QJsonDocument::fromJson(data);
            QJsonObject cameraJson = doc.object();
            
            // Load position from array
            QJsonArray positionArray = cameraJson["position"].toArray();
            qglviewer::Vec position;
            position.x = positionArray[0].toDouble();
            position.y = positionArray[1].toDouble();
            position.z = positionArray[2].toDouble();
            
            // Load orientation from array  
            QJsonArray orientationArray = cameraJson["orientation"].toArray();
            qglviewer::Quaternion orientation;
            orientation[0] = orientationArray[0].toDouble();
            orientation[1] = orientationArray[1].toDouble();
            orientation[2] = orientationArray[2].toDouble();
            orientation[3] = orientationArray[3].toDouble();
            
            // Set camera position and orientation
            m_viewer->camera()->setPosition(position);
            m_viewer->camera()->setOrientation(orientation);
            
            file.close();
        }
    }
	m_viewer->update();
}

void BroxyApp::saveScreenshot() {
	QImage screenshot = m_viewer->grabFrameBuffer();
	QString filename = QFileDialog::getSaveFileName(m_mainWindow, "Save Screenshot", QDir::currentPath() + "/screenshot.png", tr("PNG files (*.png)"));
	if (!filename.isNull()) {
		screenshot.save(filename, "PNG");
	}
}

BroxyApp::BroxyApp (int & argc, char ** argv) : QApplication (argc, argv) {
	m_mainWindow = new QMainWindow;

	QActionGroup * fileActionGroup = new QActionGroup (m_mainWindow);
	QAction * fileOpenAction = new QAction ("Open Mesh", this);
	fileOpenAction->setShortcut (tr ("Ctrl+O"));
	connect (fileOpenAction, SIGNAL (triggered()), this, SLOT (openMesh()));
	fileActionGroup->addAction (fileOpenAction);
    QAction * gsOpenAction = new QAction ("Open 3DGS", this);
    gsOpenAction->setShortcut (tr ("Ctrl+G"));
    connect (gsOpenAction, SIGNAL (triggered()), this, SLOT (open3DGS()));
    fileActionGroup->addAction (gsOpenAction);
	QAction * fileSaveAction = new QAction ("Save Proxy", this);
	fileSaveAction->setShortcut (tr ("Ctrl+S"));
	connect (fileSaveAction, SIGNAL (triggered()), this, SLOT (saveProxy()));
	fileActionGroup->addAction (fileSaveAction);

	QAction * fileQuitAction = new QAction ("Quit", m_mainWindow);
	fileQuitAction->setShortcut (tr ("Ctrl+Q"));
	connect (fileQuitAction, SIGNAL (triggered()) , qApp, SLOT (closeAllWindows()));
	fileActionGroup->addAction (fileQuitAction);
	
	QMenu * fileMenu = m_mainWindow->menuBar ()->addMenu (tr ("File"));
	fileMenu->addActions (fileActionGroup->actions ());
	
	m_viewer = new BroxyViewer;
	
	m_mainWindow->setCentralWidget (m_viewer);
	m_mainWindow->setMinimumSize (1920, 1080);
	m_mainWindow->setMinimumSize (1000, 1000);
	m_mainWindow->setMinimumSize (1000, 500);
	m_mainWindow->setWindowTitle (argv[0]);

	// Create Dock
	m_dock = new QDockWidget (m_mainWindow);	
	m_dock->setAllowedAreas (Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);

	// Create Parameters Widget
	m_paramsWidget = new QWidget (m_dock);
	QVBoxLayout * paramsLayout = new QVBoxLayout ();
	m_paramsWidget->setLayout (paramsLayout);

	// Camera Control

	QGroupBox * camGroupBox = new QGroupBox ("Camera", m_paramsWidget);
	QVBoxLayout * camLayout = new QVBoxLayout;
	camGroupBox->setLayout (camLayout);
	QPushButton * saveCamButton = new QPushButton ("Save Camera", camGroupBox);
	connect (saveCamButton, SIGNAL (pressed ()), this, SLOT (saveCamera ()));
	camLayout->addWidget (saveCamButton);
	QPushButton * loadCamButton = new QPushButton ("Load Camera", camGroupBox);
	connect (loadCamButton, SIGNAL (pressed ()), this, SLOT (loadCamera ()));
	camLayout->addWidget (loadCamButton);
	QPushButton * screenshotButton = new QPushButton ("Save Screenshot", camGroupBox);
	connect (screenshotButton, SIGNAL (pressed ()), this, SLOT (saveScreenshot ()));
	camLayout->addWidget (screenshotButton);
	paramsLayout->addWidget (camGroupBox);

	// Closing Control

	QGroupBox * closingGroupBox = new QGroupBox ("Morphological Proxy", m_paramsWidget);
	QVBoxLayout * closingLayout = new QVBoxLayout;
	closingGroupBox->setLayout (closingLayout);
	QCheckBox * isAsymmetricClosingCheckBox = new QCheckBox ("Asymmetric", closingGroupBox);
	connect (isAsymmetricClosingCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleAsymmetricClosing (bool)));
	closingLayout->addWidget (isAsymmetricClosingCheckBox);

	QGroupBox * scaleFieldGroupBox = new QGroupBox ("Scale Field", closingGroupBox);
	QHBoxLayout * scaleFieldLayout = new QHBoxLayout;
	scaleFieldGroupBox->setLayout (scaleFieldLayout);
	QPushButton * increaseScaleFieldButton = new QPushButton ("Increase Base", scaleFieldGroupBox);
	connect (increaseScaleFieldButton, SIGNAL (pressed ()), m_viewer, SLOT (increaseBaseScale ()));
	scaleFieldLayout->addWidget (increaseScaleFieldButton);
	QPushButton * decreaseScaleFieldButton = new QPushButton ("Decrease Base", scaleFieldGroupBox);
	connect (decreaseScaleFieldButton, SIGNAL (pressed ()), m_viewer, SLOT (decreaseBaseScale ()));
	scaleFieldLayout->addWidget (decreaseScaleFieldButton);
	scaleFieldLayout->addStretch (0);
	closingLayout->addWidget (scaleFieldGroupBox);

	QGroupBox * rotationFieldGroupBox = new QGroupBox ("Rotation Field", closingGroupBox);
	QHBoxLayout * rotationFieldLayout = new QHBoxLayout;
	rotationFieldGroupBox->setLayout (rotationFieldLayout);
	QCheckBox * useRotationFieldCheckBox = new QCheckBox ("Active", rotationFieldGroupBox);
	connect (useRotationFieldCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleRotationField (bool)));
	rotationFieldLayout->addWidget (useRotationFieldCheckBox);
	m_symmetryComboBox = new QComboBox (rotationFieldGroupBox);
	m_symmetryComboBox->addItem ("Full sym. group");
	m_symmetryComboBox->addItem ("No sym. group");
	rotationFieldLayout->addWidget (m_symmetryComboBox);
	QPushButton * generateFrameFieldButton = new QPushButton ("Generate", rotationFieldGroupBox);
	connect (generateFrameFieldButton, SIGNAL (clicked ()), this, SLOT (generateFrameField ()));
	rotationFieldLayout->addWidget (generateFrameFieldButton);
	rotationFieldLayout->addStretch (0);
	closingLayout->addWidget (rotationFieldGroupBox);
	closingLayout->addStretch (0);
	
	paramsLayout->addWidget (closingGroupBox);

	// Meshing Control 

	QHBoxLayout * remeshingLayout = new QHBoxLayout;
	m_proxyMeshingGroupBox = new QGroupBox (tr ("Proxy Mesh"), m_paramsWidget);
	m_proxyMeshingGroupBox->setLayout (remeshingLayout);
	m_remeshingComboBox = new QComboBox (m_proxyMeshingGroupBox);
	m_remeshingComboBox->addItem ("Morphological");
	m_remeshingComboBox->addItem ("Raw");
	remeshingLayout->addWidget (m_remeshingComboBox);
	QPushButton * remeshButton = new QPushButton ("Generate", m_proxyMeshingGroupBox);
	connect (remeshButton, SIGNAL (clicked ()), this, SLOT (remesh ()));
	remeshingLayout->addWidget (remeshButton);
	remeshingLayout->addStretch (0);
	
	paramsLayout->addWidget (m_proxyMeshingGroupBox);

	// Display parameters

	QGroupBox * displayGroupBox = new QGroupBox (tr ("Display"), m_paramsWidget);
	QVBoxLayout * displayLayout = new QVBoxLayout;
	displayGroupBox->setLayout (displayLayout);

	QCheckBox * isDisplaySmoothShadingCheckBox  = new QCheckBox (tr ("Smooth shading"), displayGroupBox);
	isDisplaySmoothShadingCheckBox->setChecked (m_viewer->isDisplaySmoothShading ());
	connect (isDisplaySmoothShadingCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleDisplaySmoothShading (bool)));
	displayLayout->addWidget (isDisplaySmoothShadingCheckBox);

	QGroupBox * displayInputMeshGroupBox = new QGroupBox ("Input Mesh", displayGroupBox);
	QHBoxLayout * displayInputMeshLayout = new QHBoxLayout;
	displayInputMeshGroupBox->setLayout (displayInputMeshLayout);
	QCheckBox * isDisplayInputMeshCheckBox = new QCheckBox (tr ("Active"), displayInputMeshGroupBox);
	isDisplayInputMeshCheckBox->setChecked (m_viewer->isDisplayInputMesh ());
	connect (isDisplayInputMeshCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleDisplayInputMesh (bool)));
	displayInputMeshLayout->addWidget (isDisplayInputMeshCheckBox);
	QCheckBox * isDisplayTransparentInputMeshCheckBox = new QCheckBox (tr ("Transparent"), displayInputMeshGroupBox);
	isDisplayTransparentInputMeshCheckBox->setChecked (m_viewer->isDisplayTransparentInputMesh ());
	connect (isDisplayTransparentInputMeshCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleDisplayTransparentInputMesh (bool)));
	displayInputMeshLayout->addWidget (isDisplayTransparentInputMeshCheckBox);
	displayInputMeshLayout->addStretch (0);
	displayLayout->addWidget (displayInputMeshGroupBox);
	
	QGroupBox * displayMorphoProxyGroupBox = new QGroupBox ("Morphological Proxy", displayGroupBox);
	QHBoxLayout * displayMorphoProxyLayout = new QHBoxLayout;
	displayMorphoProxyGroupBox->setLayout (displayMorphoProxyLayout);
	QCheckBox * isDisplayProxyCheckBox = new QCheckBox (tr ("Active"), displayMorphoProxyGroupBox);
	isDisplayProxyCheckBox->setChecked (m_viewer->isDisplayProxy ());
	connect (isDisplayProxyCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleDisplayProxy (bool)));
	displayMorphoProxyLayout->addWidget (isDisplayProxyCheckBox);
	QCheckBox * isDisplayTransparentProxyCheckBox = new QCheckBox (tr ("Transparent"), displayMorphoProxyGroupBox);
	isDisplayTransparentProxyCheckBox->setChecked (m_viewer->isDisplayTransparentProxy ());
	connect (isDisplayTransparentProxyCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleDisplayTransparentProxy (bool)));
	displayMorphoProxyLayout->addWidget (isDisplayTransparentProxyCheckBox);
	displayMorphoProxyLayout->addStretch (0);
	displayLayout->addWidget (displayMorphoProxyGroupBox);

	QGroupBox * displayProxyMeshGroupBox = new QGroupBox ("Proxy Mesh", displayGroupBox);
	QHBoxLayout * displayProxyMeshLayout = new QHBoxLayout;
	displayProxyMeshGroupBox->setLayout (displayProxyMeshLayout);
	QCheckBox * isDisplayProxyMeshCheckBox = new QCheckBox (tr ("Active"), displayProxyMeshGroupBox);
	isDisplayProxyMeshCheckBox->setChecked (m_viewer->isDisplayProxyMesh ());
	connect (isDisplayProxyMeshCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleDisplayProxyMesh (bool)));
	displayProxyMeshLayout->addWidget (isDisplayProxyMeshCheckBox);
	QCheckBox * isDisplayTransparentProxyMeshCheckBox = new QCheckBox (tr ("Transparent"), displayProxyMeshGroupBox);
	isDisplayTransparentProxyMeshCheckBox->setChecked (m_viewer->isDisplayTransparentProxyMesh ());
	connect (isDisplayTransparentProxyMeshCheckBox, SIGNAL (toggled (bool)), m_viewer, SLOT (toggleDisplayTransparentProxyMesh (bool)));
	displayProxyMeshLayout->addWidget (isDisplayTransparentProxyMeshCheckBox);
	displayProxyMeshLayout->addStretch (0);
	displayLayout->addWidget (displayProxyMeshGroupBox);
	displayLayout->addStretch (0);
	
	paramsLayout->addWidget (displayGroupBox);
	paramsLayout->addStretch (0);

	// Attach Params Widget to the Dock
	m_dock->setWidget (m_paramsWidget);
	m_mainWindow->addDockWidget (Qt::RightDockWidgetArea, m_dock);

	connect (this, SIGNAL (aboutToQuit ()), this, SLOT (clean ()));
	connect (this, SIGNAL (lastWindowClosed()), this, SLOT (quit()));
}

BroxyApp::~BroxyApp () {
	clean ();
}

int BroxyApp::run () {
	m_mainWindow->show ();
	return this->exec ();
}

void BroxyApp::openMesh () {
	QString filename = QFileDialog::getOpenFileName (m_mainWindow,
													 "Open mesh",
													 QDir::currentPath (),
													 tr ("OFF (*.off)"));
	if (!filename.isNull()) 
		m_viewer->loadMesh (filename);
}

void BroxyApp::open3DGS () {
    QString filename = QFileDialog::getExistingDirectory(m_mainWindow,
                                                     "Open 3dgs",
                                                     QDir::currentPath ()
                                                     );
    bool ok;
    QString text = QInputDialog::getText(
            m_mainWindow,
            tr("GS3D Density Threshold"), tr("GS3D Density Threshold"),
            QLineEdit::Normal, "1e-6",
            &ok
    );
    if (!ok) {
        std::cout << "debug - loading 3DGS failed - " << "QInputDialog cancelled" << std::endl;
        return;
    }
    float density = text.toFloat(&ok);
    if(!ok) {
        std::cout << "debug - loading 3DGS failed - " << "text is not valid float literal" << std::endl;
        return;
    }

    std::cout << "debug - loading 3DGS with density " << density << std::endl;

    // Select voxelization method
    QStringList methods;
    methods << "TSDF (VoxelizeConservativeTSDF)" << "Conservative (VoxelizeConservative)";
    QString method = QInputDialog::getItem(
            m_mainWindow,
            tr("Voxelization Method"), tr("Select voxelization method:"),
            methods, 0, false, &ok
    );
    if (!ok) {
        std::cout << "debug - loading 3DGS failed - " << "method selection cancelled" << std::endl;
        return;
    }
    bool use_tsdf = (method == methods[0]);

    if (!filename.isNull())
        m_viewer->load3DGS(filename, density, use_tsdf);
}

void BroxyApp::saveProxy () {
	QString filename = QFileDialog::getSaveFileName (m_mainWindow,
													 "Save proxy",
													 QDir::currentPath () + "/proxy.ply",
													 tr ("PLY Files (*.ply);;OBJ Files (*.obj);;OFF Files (*.off);;All Files (*)"));
	if (!filename.isNull()) 
		m_viewer->storeProxy (filename);
}

void BroxyApp::clean () {
	// Clean up code.
}

void BroxyApp::remesh () {
	QStringList methods;
	methods << "Standard Decimation" << "Interleaved Optimization";
	bool ok;
	QString method = QInputDialog::getItem (m_mainWindow,
	                                        tr ("Decimation Method"),
	                                        tr ("Select decimation method:"),
	                                        methods, 0, false, &ok);
	if (!ok)
		return;
	if (method == "Interleaved Optimization") {
		m_viewer->runInterleavedDecimation ();
	} else {
		QString current = m_remeshingComboBox->currentText ();
		bool onMorphoInput = true;
		if (current == "Constrained Decimation Only")
			onMorphoInput = false;
		m_viewer->featureAwareDecimation (onMorphoInput);
	}
}

void BroxyApp::generateFrameField () {
	bool updateOnly = true;
	bool useGroupSymmetry = true;
	QString current = m_symmetryComboBox->currentText ();
	if (current != "Full symmetry group") 
		useGroupSymmetry = false;
	m_viewer->generateFrameField (updateOnly, useGroupSymmetry);
}