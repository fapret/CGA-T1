// ======================================================================== //
// Copyright 2018-2021 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "SampleRenderer.h"

// our helper library for window handling
#include "owlViewer/OWLViewer.h"
#include <GL/gl.h>
#include "kdtree.h"
#include <array>
#include <iostream>
#include <chrono>
#include <ctime>

/*! \namespace osc - Optix Siggraph Course */
namespace cga {

  struct SampleWindow : public owl::viewer::OWLViewer
  {
    SampleWindow(const std::string &title,
                 const Model *model,
                 const Camera &camera,
                 const QuadLight &light,
                 const float worldScale)
      : OWLViewer(title// ,camera.from,camera.at,camera.up,worldScale
                  ),
        sample(model,light)
    {
      this->camera.setOrientation(camera.from,
                                  camera.at,
                                  camera.up,
                                  60.f);
      this->setWorldScale(worldScale);
      sample.setCamera(camera);
    }
    
    virtual void render() override
    {
      if (camera.lastModified != 0) {
        
        sample.setCamera(Camera{ camera.getFrom(),
                                 camera.getAt(),
                                 camera.getUp() });
        camera.lastModified = 0;
      }
      sample.render();
    }
    
    void resize(const vec2i &newSize) override
    {
      OWLViewer::resize(newSize);
      // fbSize = newSize;
      sample.resize(fbPointer,newSize);
    }

    void key(char key, const vec2i &where) override
    {
      if (key == 'p') {
          auto currentTime = std::chrono::system_clock::now();
          std::time_t time = std::chrono::system_clock::to_time_t(currentTime);
          char timeStr[100];
          std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H-%M-%S.png", std::localtime(&time));
          OWLViewer::screenShot(timeStr);
      }

      OWLViewer::key(key,where);
    }
    
    SampleRenderer        sample;
  };


  
  /*! main entry point to this example - initially optix, print hello
    world, then exit */
  extern "C" int main(int ac, char **av)
  {
    std::string inFileName = 
#ifdef _WIN32
#ifdef OWL_BUILDING_ALL_SAMPLES
      // on windows, when building the whole project (including the
      // samples) with VS, the executable's location is different
      // ../../project/scene.obj en visual, ./scene.obj para debug
      "./cornell-box.obj"
#else
      // on windows, visual studio creates _two_ levels of build dir
      // (x86/Release)
      "../../scene.obj"
#endif
#else
      // on linux, common practice is to have ONE level of build dir
      // (say, <project>/build/)...
      "./scene.obj"
      //      "../models/sponza.obj"
#endif
      ;
    if (ac == 2)
      inFileName = av[1];
    try {
      Model *model = loadOBJ(inFileName);
      vec3f from = model->bounds.center() + vec3f(0, 0, 12);
      vec3f at = model->bounds.center();
      Camera camera = { /*from*/from,
                        /* at */at,
                        /* up */vec3f(0.f,1.f,0.f) };


      

      //std::vector<photon> points = { photon(1.0, 2.0, 3.0, 'asfc', 'a', 'b', 23), photon(3.0, 4.0, 3.0, 'asfc', 'a', 'b', 23), photon(5.0, 6.0, 3.0, 'asfc', 'a', 'b', 23)};
      //char x = points[1].phi;

      //kdt::KDTree<photon> kdtree(points);

      // some simple, hard-coded light ... obviously, only works for sponza
      const float light_size = 0.f;
      vec3f upper = model->bounds.upper;
      QuadLight light = { /* origin */ model->bounds.center() + vec3f(0.f,upper.y - model->bounds.center().y - 1.3f,0.f),  //model->bounds.center() + vec3f(0.f,upper.y - model->bounds.center().y -1.3f,0.f),
                          /* edge 1 */ vec3f(1.f*light_size,0,0),
                          /* edge 2 */ vec3f(0,0,1.f * light_size),
                          /* power */  vec3f(10.f) };
                      
      // something approximating the scale of the world, so the
      // camera knows how much to move for any given user interaction:
      const float worldScale = length(model->bounds.span());

      SampleWindow *window = new SampleWindow("Optix 7 Course Example (on OWL)",
                                              model,camera,light,worldScale);
      window->enableFlyMode();
      
      std::cout << "Press 'A' to enable/disable accumulation/progressive refinement" << std::endl;
      std::cout << "Press 'D' to enable/disable denoising" << std::endl;
      std::cout << "Press ',' to reduce the number of paths/pixel" << std::endl;
      std::cout << "Press '.' to increase the number of paths/pixel" << std::endl;
      window->showAndRun();
      
    } catch (std::runtime_error& e) {
      std::cout << OWL_TERMINAL_RED << "FATAL ERROR: " << e.what()
                << OWL_TERMINAL_DEFAULT << std::endl;
	  std::cout << "Did you forget to copy sponza.obj and sponza.mtl into your optix7course/models directory?" << std::endl;
	  exit(1);
    }
    return 0;
  }
  
} // ::osc

