/*
 *  Created by Parag K. Mital - http://pkmital.com
 *  Contact: parag@pkmital.com
 *
 *  Copyright 2011 Parag K. Mital. All rights reserved.
 *
 *	Permission is hereby granted, free of charge, to any person
 *	obtaining a copy of this software and associated documentation
 *	files (the "Software"), to deal in the Software without
 *	restriction, including without limitation the rights to use,
 *	copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the
 *	Software is furnished to do so, subject to the following
 *	conditions:
 *
 *	The above copyright notice and this permission notice shall be
 *	included in all copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *	EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 *	OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *	NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 *	HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 *	WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *	FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 *	OTHER DEALINGS IN THE SOFTWARE.
 */

#include "testApp.h"

//--------------------------------------------------------------
testApp::~testApp(){
    delete fft;
    delete recorder;
    
    b_setup = false;
}

//--------------------------------------------------------------
void testApp::setup(){

    b_setup = false;
    
    // globals
    width = 1000;
    height = 100;
    n_sample_rate = 44100;
    n_buffer_size = 512;
    
    // reshape window size
    ofSetWindowShape(width, height * 2);
    
    // setup drawing of the waveform
    waveform.setup(0,                   // x position to draw
                   0,                   // y
                   width,               // ui drawing width
                   height,              // height
                   n_buffer_size,       // each audio callback frame
                   1.0);                // drawing resolution (0.5 - 2.0)
    waveform.setLoopRegion(true);

    // load up an audiofile
    string input_file(ofToDataPath("target.wav"));
    if(!waveform.loadFile(input_file))
    {
        printf("[ERROR] File %s not found\n", input_file.c_str());
        OF_EXIT_APP(0);
    }
    
    // load auto-encoder stuff
    hb.load(ofToDataPath("hb.csv"));
    vb.load(ofToDataPath("vb.csv"));
    W.load(ofToDataPath("W.csv"));
    W_prime.load(ofToDataPath("Wprime.csv"));
    
    // create a layer for the activations and the interactive weights on them
    n_neurons = hb.rows;
    activationLayer = pkm::Mat(1, n_neurons);
    weights = pkm::Mat(1, n_neurons, 1.0f);
    
//    // print the autoencoder data
//    W.printAbbrev();
//    W_prime.printAbbrev();
//    hb.printAbbrev();
//    vb.printAbbrev();
    
    // transpose biases
    hb.setTranspose();
    vb.setTranspose();
    
    // setup ring buffer
    n_fft_size = 2048;
    fft_frame = pkm::Mat(1, n_fft_size, 0.0f);
    overlap_frame = pkm::Mat(1, n_fft_size - n_buffer_size, 0.0f);
    recorder = new pkmCircularRecorder(n_fft_size, n_buffer_size);
    
    // setup fft stuff
    fft = new pkmFFT(n_fft_size);
    mags = pkm::Mat(1, n_fft_size/2 + 1);
    phases = pkm::Mat(1, n_fft_size/2 + 1);
    
    // setup audio callbacks
    ofSoundStreamSetup(1,               // output channels
                       0,               // input channels
                       n_sample_rate,   // samples per second
                       n_buffer_size,   // samples per audio call back
                       4);              // number of audio callback buffers
    
    initializeGUI();
    
    b_setup = true;
}

//--------------------------------------------------------------
void testApp::update(){
    if(!b_setup)
        return;
    
    waveform.updateWaveform();
    gui->update();
    
}

//--------------------------------------------------------------
void testApp::draw(){
    if(!b_setup)
        return;
    
    waveform.draw();
    gui->draw();
}


#pragma mark gui_callbacks

//--------------------------------------------------------------
void testApp::initializeGUI(){
    gui = new ofxUICanvas(0,height,width,height);
    
    if (!gui->setFont(ofToDataPath("helvetica-light-normal.ttf")))
    {
        cout << "Couldn't find font!" << endl;
        OF_EXIT_APP(0);
    }
    
    int slider_width = width / n_neurons;
    for (int i = 0; i < n_neurons; i++) {
        gui->addWidget(new ofxUISlider(3 + i * slider_width, 0, slider_width, height * 0.9, -1.0, 1.0, 1.0, ofToString(i)));
        gui->getWidget(ofToString(i))->setDrawOutlineHighLight(true);
    }
    
    ofAddListener(gui->newGUIEvent, this, &testApp::guiEvent);
    
    gui->enable();
    gui->toggleVisible();
}

//--------------------------------------------------------------
void testApp::guiEvent(ofxUIEventArgs &e){
    string name = e.widget->getName();
    ofxUISlider *slider = (ofxUISlider *)e.widget;
    weights[ofToInt(name)] = slider->getValue();
    weights.print();
}


#pragma mark audio_callbacks

//--------------------------------------------------------------
void testApp::audioIn(float *buf, int size, int ch){
    vDSP_vclr(buf, 1, size);
}

//--------------------------------------------------------------
void testApp::audioOut(float *buf, int size, int ch){
    // get current buffer
    waveform.readFrameAndIncrement(buf);
    recorder->insertFrame(buf);
    recorder->copyAlignedData(fft_frame.data);
    vDSP_vclr(buf, 1, size);
    
    // apply fft
    bool b_apply_window = true;
    fft->forward(0, fft_frame.data, mags.data, phases.data, b_apply_window);
    vDSP_vclr(fft_frame.data, 1, n_fft_size);
    
    // into activation layer
    mags.GEMM(W, activationLayer);
    activationLayer.add(hb);
    
//    // sigmoid function
//    activationLayer.multiply(-1.0f);
//    activationLayer.exp();
//    activationLayer.add(1.0);
//    activationLayer.pow(-1.0);
    
    // interactive control
    activationLayer = activationLayer.multiply(weights);
    
    // back out
    activationLayer.GEMM(W_prime, mags);
    mags.add(vb);
    
    // add previous overlap
    vDSP_vadd(fft_frame.data, 1, overlap_frame.data, 1, fft_frame.data, 1, n_fft_size - size);
    
    // do inverse
    fft->inverse(0, fft_frame.data, mags.data, phases.data, b_apply_window);
    cblas_scopy(size, fft_frame.data, 1, buf, 1);
    
    // store overlap
    cblas_scopy(n_fft_size - size, fft_frame.data + size, 1, overlap_frame.data, 1);
}

#pragma mark key_callbacks

//--------------------------------------------------------------
void testApp::keyPressed(int key){

}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

#pragma mark mouse_callbacks

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

    waveform.mouseMoved(x, y);
    gui->mouseMoved(x, y);
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

    waveform.mouseDragged(x, y);
    gui->mousePressed(x, y, button);
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

    waveform.mousePressed(x, y);
    gui->mousePressed(x, y, button);
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

    waveform.mouseReleased(x, y);
}

#pragma mark window_ui_callbacks

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}
