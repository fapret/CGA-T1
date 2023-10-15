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

namespace cga {
  using namespace owl;

  // for this simple example, we have a single ray type
  enum { RADIANCE_RAY_TYPE=0, SHADOW_RAY_TYPE, RAY_TYPE_COUNT };

  struct TriangleMeshSBTData {
    vec3f  color;
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
      double x;
      double y;
      double z;

      double& operator[](int index) {
          switch (index) {
          case 0:
              return x;
          case 1:
              return y;
          case 2:
              return z;
          }
          throw std::out_of_range("Index out of range");
      };
      __host__ __device__ Photon() {
          this->x = 0;
          this->y = 0;
          this->z = 0;
          *this->p = 'd';
          this->phi = 'd';
          this->theta = 'd';
          this->flag = 'd';
      };
      __host__ __device__ Photon(double x, double y, double z, char p, char phi, char theta, short flag)
      {
          this->x = x;
          this->y = y;
          this->z = z;
          *this->p = p;
          this->phi = phi;
          this->theta = theta;
          this->flag = flag;
      }
  };

  
  
  struct LaunchParams
  {
    int numPixelSamples = 1;
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
  
  };

} // ::osc
