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

#pragma once

#include "Model.h"
#include "owl/owl.h"
#include "kdtree.h"
#include <array>

#include <iostream>
#include <fstream>

namespace cga {
  using namespace owl;

  // for this simple example, we have a single ray type
  enum { RADIANCE_RAY_TYPE=0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

  struct TriangleMeshSBTData {
    vec3f  color;
    vec3f  specular;
    vec3f *vertex;
    vec3f *normal;
    vec2f *texcoord;
    vec3i *index;
    int    hasTexture;
    cudaTextureObject_t texture;
  };


  struct Photon {
      static const int DIM = 3;
      char p[4];
      char phi, theta;
      short flag;
      int threadId;
      vec3f dir;
      vec3f position = vec3f(0,0,0);
      vec3f color;
      int index = -1;
      float power;
      int timesBounced;

      float& operator[](int index) {
          switch (index) {
          case 0:
              return position.x;
          case 1:
              return position.y;
          case 2:
              return position.z;
          }
          throw std::out_of_range("Index out of range");
      };
      __host__ __device__ Photon() {
          *this->p = 'd';
          this->phi = 'd';
          this->theta = 'd';
          this->flag = 'd';
      };
      __host__ __device__ Photon(float x, float y, float z, char p, char phi, char theta, short flag)
      {
          
          this->position = vec3f(x, y, z);
          *this->p = p;
          this->phi = phi;
          this->theta = theta;
          this->flag = flag;
      }
  };

  
  
  struct LaunchParams
  {
    int numPixelSamples = 1;
    int numOfPhotons = 170;
    int numOfBounces = 4 ;
    int sphereRadius = 5;
    bool photonMap = true;
    bool rayTrace = true;

    struct {
      int       frameID = 0;
      // the *final* frame buffer, after accum
      uint32_t    *fbFinal;
      // the color buffer, for accum buffering
      float4      *fbColor;
      // float4   *colorBuffer;
      // float4   *normalBuffer;
      // float4   *albedoBuffer;
      
      /*! the size of the frame buffer to render */
      vec2i     fbSize;
    } frame;
    
    struct {
      vec3f position;
      vec3f direction;
      vec3f horizontal;
      vec3f vertical;
    } camera;

    struct {
      vec3f origin, du, dv, power;
    } light;

    OptixTraversableHandle traversable;

    Photon* photonArray;
    LaunchParams() {
        std::fstream my_file;
        my_file.open("C:/Users/ferna/Documents/Facultad/GRAFA/CGA-T1/project/config.txt", std::ios::in);
        if (!my_file) {
            std::cout << "No such file";
        }
        else {
            std::string line;
            std::string key;
            std::string value;

            

            while (std::getline(my_file, line)) {
                // Process each line here
                std::stringstream ss(line);
                std::getline(ss, key, '=');
                std::getline(ss, value);

                key.erase(0, key.find_first_not_of(" "));
                key.erase(key.find_last_not_of(" ") + 1);

                value.erase(0, value.find_first_not_of(" "));
                value.erase(value.find_last_not_of(" ") + 1);

                if (key == "numOfPhotons") {
                    numOfPhotons = std::stoi(value);
                }
                if (key == "numOfBounces") {
                    numOfBounces = std::stoi(value);
                }
                if (key == "sphereRadius") {
                    sphereRadius = std::stoi(value);
                }
                if (key == "photonMap") {
                    if (value == "true") {
                        photonMap = true;
                    }
                    else {
                        photonMap = false;
                    }
                }
                if (key == "rayTrace") {
                    if (value == "true") {
                        rayTrace = true;
                    }
                    else {
                        rayTrace = false;
                    }
                }
                std::cout << line << std::endl; // Print the line to the console as an example
            }

        }
        my_file.close();
    }
  
  };

} // ::osc
