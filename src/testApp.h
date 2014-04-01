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

#pragma once

#include "ofMain.h"
#include "pkmMatrix.h"
#include "pkmFFT.h"
#include "pkmEXTAudioFileReader.h"
#include "pkmAudioWaveform.h"
#include "ofxUI.h"

//--------------------------------------------------------------
class testApp : public ofBaseApp{

public:
    //--------------------------------------------------------------
    void setup();
    void update();
    void draw();
    ~testApp();
    
    //--------------------------------------------------------------
    void audioIn(float *buf, int size, int ch);
    void audioOut(float *buf, int size, int ch);
    
    //--------------------------------------------------------------
    void initializeGUI();
    void guiEvent(ofxUIEventArgs &e);
    
    //--------------------------------------------------------------
    void keyPressed(int key);
    void keyReleased(int key);
    
    //--------------------------------------------------------------
    void mouseMoved(int x, int y );
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    
    //--------------------------------------------------------------
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
private:
    //--------------------------------------------------------------
    pkmAudioWaveform waveform;
    
    //--------------------------------------------------------------
    pkmFFT *fft;
    pkm::Mat mags, phases;
    
    //--------------------------------------------------------------
    pkm::Mat hb, hbt, vb, W, W_prime;
    pkm::Mat weights;
    pkm::Mat activationLayer;
    int n_neurons;
    
    //--------------------------------------------------------------
    ofxUICanvas *gui;
    
    //--------------------------------------------------------------
    int width, height;
    int n_channels, n_buffer_size, n_sample_rate;
    
    bool b_setup;
};
