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

#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include <owl/common/math/random.h>
#include <vector>
#include <math.h>
#include <cmath>



using namespace cga;

#define NUM_LIGHT_SAMPLES 4

namespace cga {

  typedef owl::common::LCG<16> Random;
  
  /*! launch parameters in constant memory, filled in by optix upon
      optixLaunch (this gets filled in from the buffer we pass to
      optixLaunch) */
  extern "C" __constant__ LaunchParams optixLaunchParams;

  static __forceinline__ __device__
  void calculateNewVector(vec3f& normalVector, float polarAngle, float azimuthalAngle, vec3f& newVector) {
      float r = sqrtf(normalVector.x * normalVector.x + normalVector.y * normalVector.y + normalVector.z * normalVector.z);
      float theta = polarAngle;
      float phi = azimuthalAngle;

      newVector.x = r * sin(theta) * cos(phi);
      newVector.y = r * sin(theta) * sin(phi);
      newVector.z = r * cos(theta);

      if (dot(newVector, normalVector) < 0.0f) {
          newVector = -newVector;
      }
  }


  /*! per-ray data now captures random number generator, so programs
      can access RNG state */
  struct PRD {
    Random random;
    vec3f  pixelColor;
    bool   isPhoton;
    Photon photon;
    vec3f  pixelNormal;
    vec3f  pixelAlbedo;
    vec3f  position = vec3f(0, 0, 0);
  };

  
  static __forceinline__ __device__
  void *unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }

  static __forceinline__ __device__
  void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }

  template<typename T>
  static __forceinline__ __device__ T *getPRD()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
  }


  
  //------------------------------------------------------------------------------
  // closest hit and anyhit programs for radiance-type rays.
  //
  // Note eventually we will have to create one pair of those for each
  // ray type and each geometry type we want to render; but this
  // simple example doesn't use any actual geometries yet, so we only
  // create a single, dummy, set of them (we do have to have at least
  // one group of them to set up the SBT)
  //------------------------------------------------------------------------------
  
  extern "C" __global__ void __closesthit__shadow()
  {
    return;
    /* not going to be used ... */
  }
  
  extern "C" __global__ void __closesthit__radiance()
  {
    const TriangleMeshSBTData &sbtData
      = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    // ------------------------------------------------------------------
    // gather some basic hit information
    // ------------------------------------------------------------------
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int   primID = optixGetPrimitiveIndex();
    const vec3i index = sbtData.index[primID];

    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // ------------------------------------------------------------------
    // compute normal, using either shading normal (if avail), or
    // geometry normal (fallback)
    // ------------------------------------------------------------------
    const vec3f& A = sbtData.vertex[index.x];
    const vec3f& B = sbtData.vertex[index.y];
    const vec3f& C = sbtData.vertex[index.z];

    vec3f Ng = cross(B - A, C - A);
    vec3f Ns = (sbtData.normal)
        ? ((1.f - u - v) * sbtData.normal[index.x]
            + u * sbtData.normal[index.y]
            + v * sbtData.normal[index.z])
        : Ng;

    // ------------------------------------------------------------------
    // face-forward and normalize normals
    // ------------------------------------------------------------------
    const vec3f rayDir = optixGetWorldRayDirection();

    if (dot(rayDir, Ng) > 0.f) Ng = -Ng;
    Ng = normalize(Ng);

    if (dot(Ng, Ns) < 0.f)
        Ns -= 2.f * dot(Ng, Ns) * Ng;
    Ns = normalize(Ns);


    // ------------------------------------------------------------------
        // compute diffuse material color, including diffuse texture, if
        // available
        // ------------------------------------------------------------------
    vec3f diffuseColor = sbtData.color;

    if (sbtData.hasTexture && sbtData.texcoord) {
        const vec2f tc
            = (1.f - u - v) * sbtData.texcoord[index.x]
            + u * sbtData.texcoord[index.y]
            + v * sbtData.texcoord[index.z];

        vec4f fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor *= (vec3f)fromTexture;
    }

    const vec3f surfPos
        = (1.f - u - v) * sbtData.vertex[index.x]
        + u * sbtData.vertex[index.y]
        + v * sbtData.vertex[index.z];

    prd.position = surfPos;

    if (!prd.isPhoton) {
        // start with some ambient term
        vec3f pixelColor = (0.1f + 0.2f * fabsf(dot(Ns, rayDir))) * diffuseColor;

        // ------------------------------------------------------------------
        // compute shadow
        // ------------------------------------------------------------------
        

        const int numLightSamples = NUM_LIGHT_SAMPLES;
        for (int lightSampleID = 0; lightSampleID < numLightSamples; lightSampleID++) {
            // produce random light sample
            const vec3f lightPos
                = optixLaunchParams.light.origin
                + prd.random() * optixLaunchParams.light.du
                + prd.random() * optixLaunchParams.light.dv;
            vec3f lightDir = lightPos - surfPos;
            float lightDist = length(lightDir);
            lightDir = normalize(lightDir);

            // trace shadow ray:
            const float NdotL = dot(lightDir, Ns);
            if (NdotL >= 0.f) {
                vec3f lightVisibility = 0.f;
                // the values we store the PRD pointer in:
                uint32_t u0, u1;
                packPointer(&lightVisibility, u0, u1);
                optixTrace(optixLaunchParams.traversable,
                    surfPos + 1e-3f * Ng,
                    lightDir,
                    1e-3f,      // tmin
                    lightDist * (1.f - 1e-3f),  // tmax
                    0.0f,       // rayTime
                    OptixVisibilityMask(255),
                    // For shadow rays: skip any/closest hit shaders and terminate on first
                    // intersection with anything. The miss shader is used to mark if the
                    // light was visible.
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT
                    | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                    SHADOW_RAY_TYPE,            // SBT offset
                    RAY_TYPE_COUNT,               // SBT stride
                    SHADOW_RAY_TYPE,            // missSBTIndex 
                    u0, u1);
                pixelColor
                    += lightVisibility
                    * optixLaunchParams.light.power
                    * diffuseColor
                    * (NdotL / (lightDist * lightDist * numLightSamples));
            }
        }
        prd.pixelNormal = Ns;
        prd.pixelAlbedo = diffuseColor;
        prd.pixelColor = pixelColor;
    }
    else {
        // es un foton que paga en una superfice, tenemos que meterlo en el kdtree
        // tenemos que rebotarlo
        // guardo el foton en el buffer, se guarda el foton que llega con el color que trae
        prd.photon.position = surfPos;

        //if (prd.photon.timesBounced > 1) {
        //    printf("Bounced Times: %d\n", prd.photon.timesBounced);
       // }

        const vec3f lightPos
            = optixLaunchParams.light.origin
            + prd.random() * optixLaunchParams.light.du
            + prd.random() * optixLaunchParams.light.dv;

        vec3f lightDir = lightPos - surfPos;
        lightDir = normalize(lightDir);

        
        
        if (prd.photon.timesBounced + 1 > optixLaunchParams.numOfBounces) {
            return;
        }
        
        float numeradorD;
        if ((diffuseColor[0] * prd.photon.color[0]) > (diffuseColor[1] * prd.photon.color[1])) {
            numeradorD = diffuseColor[0] * prd.photon.color[0];
        }
        else {
            numeradorD = diffuseColor[1] * prd.photon.color[1];
        }
        if (numeradorD < (diffuseColor[2] * prd.photon.color[2])) {
            numeradorD = diffuseColor[2] * prd.photon.color[2];
        }

        float numeradorS;
        if ((sbtData.specular[0] * prd.photon.color[0]) > (sbtData.specular[1] * prd.photon.color[1])) {
            numeradorS = sbtData.specular[0] * prd.photon.color[0];
        }
        else {
            numeradorS = sbtData.specular[1] * prd.photon.color[1];
        }
        if (numeradorS < (sbtData.specular[2] * prd.photon.color[2])) {
            numeradorS = sbtData.specular[2] * prd.photon.color[2];
        }

        float divisor;
        if (prd.photon.color[0] > prd.photon.color[1]) {
            divisor = prd.photon.color[0];
        }
        else {
            divisor = prd.photon.color[1];
        }
        if (divisor < prd.photon.color[2]) {
            divisor = prd.photon.color[2];
        }

        float Pd = numeradorD / divisor;
        float Ps = numeradorS / divisor;

        float randomNum = prd.random();
 
        if (randomNum > Pd + Ps) {
            // absorbido
            return;
        }

        PRD prd_bouced;
        uint32_t u_0, u_1;
        packPointer(&prd_bouced, u_0, u_1);
        prd_bouced.isPhoton = true;
        vec3f bounceDir = vec3f(0, 0, 0);

        Photon bouncedPhoton = Photon();
        bouncedPhoton.index = prd.photon.index + 1;
        bouncedPhoton.timesBounced = prd.photon.timesBounced + 1;
        
        bouncedPhoton.threadId = prd.photon.threadId;
        vec3f bouncedPhotonColor = vec3f(0.f,0.f,0.f);

        if (randomNum <= Pd) {
            if (prd.photon.timesBounced >= 0) {
                optixLaunchParams.photonArray[prd.photon.index] = prd.photon;
            }
            // es difusa

            // Ns es la normal normalizada en el punto de impacto
            // photon.dir es la direccion de impacto normalizada

            float alpha = prd.random() * 90;
            float phi = prd.random() * 360;

            float alpha_rad = alpha * M_PI / 180.0;
            float phi_rad = phi * M_PI / 180.0;

            calculateNewVector(Ns, alpha_rad, phi_rad, bounceDir);

            //Color_difuso_resultante = Color_del_fotón * Color_de_la_superficie * (cos(theta)), donde "theta" es el ángulo entre la dirección de la luz incidente y la normal de la superficie.
            // Ns es la normal de la superfice

            bouncedPhotonColor = vec3f(prd.photon.color[0] * diffuseColor[0] / Pd, prd.photon.color[1] * diffuseColor[1] / Pd, prd.photon.color[2] * diffuseColor[2] / Pd) ;
            //bouncedPhotonColor = prd.photon.color * diffuseColor * dot(Ns, lightDir);

            //printf("Bounced Color %f - %f - %f\n", bouncedPhotonColor.x, bouncedPhotonColor.y, bouncedPhotonColor.z);
        }
        else if (randomNum <= Pd + Ps) {
            // es especular
            
            bounceDir = prd.photon.dir  - 2.f * dot(prd.photon.dir, Ns) * Ns;
            vec3f cameraDir = optixLaunchParams.camera.position - surfPos;
            cameraDir = normalize(cameraDir);
            vec3f colorEspecular = sbtData.specular * dot(bounceDir, cameraDir);
            bouncedPhotonColor = vec3f(prd.photon.color[0] * diffuseColor[0] / Ps, prd.photon.color[1] * diffuseColor[1] / Ps, prd.photon.color[2] * diffuseColor[2] / Ps);
            //printf("Bounced dir %f - %f - %f\n", surfPos.x, surfPos.y, surfPos.z);
        }
        
        bounceDir = normalize(bounceDir);
        bouncedPhoton.dir = bounceDir;
        bouncedPhoton.color = bouncedPhotonColor;
        prd_bouced.pixelColor = bouncedPhoton.color;
        prd_bouced.photon = bouncedPhoton;

        optixTrace(optixLaunchParams.traversable,
            surfPos + 1e-3f * bounceDir,
            bounceDir,
            0.f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
            RADIANCE_RAY_TYPE,            // SBT offset
            RAY_TYPE_COUNT,               // SBT stride
            RADIANCE_RAY_TYPE,            // missSBTIndex 
            u_0, u_1);
    }
  }
  
  extern "C" __global__ void __anyhit__radiance()
  { /*! for this simple example, this will remain empty */ }

  extern "C" __global__ void __anyhit__shadow()
  { /*! not going to be used */ }
  
  //------------------------------------------------------------------------------
  // miss program that gets called for any ray that did not have a
  // valid intersection
  //
  // as with the anyhit/closest hit programs, in this example we only
  // need to have _some_ dummy function to set up a valid SBT
  // ------------------------------------------------------------------------------
  
  extern "C" __global__ void __miss__radiance()
  {
    PRD &prd = *getPRD<PRD>();
     // set to constant white as background color
     prd.pixelColor = vec3f(0.15f);
  }

  extern "C" __global__ void __miss__shadow()
  {
    // we didn't hit anything, so the light is visible
    vec3f &prd = *(vec3f*)getPRD<vec3f>();
    prd = vec3f(1.f);
  }


  
  extern "C" __global__ void __raygen__emitPhoton()
  {
      const int ix = optixGetLaunchIndex().x;
      const int iy = optixGetLaunchIndex().y;
      PRD prd_photon;
      prd_photon.isPhoton = true;
      prd_photon.random.init(ix + optixLaunchParams.frame.fbSize.x * iy,
          optixLaunchParams.frame.frameID);
      prd_photon.pixelColor = vec3f(0.f);
      const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.fbSize.x;

      int numPixelSamples = optixLaunchParams.numPixelSamples;

      // the values we store the PRD pointer in:
      uint32_t u_0, u_1;
      packPointer(&prd_photon, u_0, u_1);
      
      int threadId_x = threadIdx.x;
      int threadId_y = threadIdx.y;

      int threadId = threadId_x + (threadId_y * 1200);


      const vec3f lightPos
          = optixLaunchParams.light.origin;
          //+ prd_photon.random() * optixLaunchParams.light.du
          //+ prd_photon.random() * optixLaunchParams.light.dv;
      
      const auto& camera = optixLaunchParams.camera;
      

      vec2f screen(vec2f(ix + prd_photon.random(), iy + prd_photon.random())
          / vec2f(optixLaunchParams.frame.fbSize));


      vec3f centroEscena = vec3f(-107.297424, 37.1369781, -127.297424);
      float y_max = 580;
      float y_min = -400;
      float z_max = 430;
      float z_min = -670;

      for (int i = 0; i < optixLaunchParams.numOfPhotons; i++) {
          vec3f pixelColor = 0.f;
          Photon photon = Photon(lightPos.x, lightPos.y, lightPos.z, 'a', 'a', 'a', 3);
          photon.color = vec3f(1.f);
          photon.index = threadId * optixLaunchParams.numOfPhotons * (optixLaunchParams.numOfBounces + 1) + i * (optixLaunchParams.numOfBounces + 1);
          photon.threadId = threadId;
          //printf("Photon index %d\n", i);
          photon.timesBounced = 0;


          float y = (prd_photon.random() * (y_max - y_min)) + y_min;
          float z = (prd_photon.random() * (z_max - z_min)) + z_min;

          vec3f offset = vec3f(0, y, z);

          vec3f rayDir = centroEscena + offset - lightPos;
          rayDir = normalize(rayDir);


          
          //vec3f rayDir = normalize(camera.direction
          //    + (screen.x - 0.5f) * camera.horizontal
          //    + (screen.y - 0.5f) * camera.vertical);

          photon.dir = rayDir;
          prd_photon.photon = photon;

          optixTrace(optixLaunchParams.traversable,
              lightPos,
              rayDir,
              0.f,    // tmin
              1e20f,  // tmax
              0.0f,   // rayTime
              OptixVisibilityMask(255),
              OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
              RADIANCE_RAY_TYPE,            // SBT offset
              RAY_TYPE_COUNT,               // SBT stride
              RADIANCE_RAY_TYPE,            // missSBTIndex 
              u_0, u_1);
      }
  }



  //------------------------------------------------------------------------------
  // ray gen program - the actual rendering happens in here
  //------------------------------------------------------------------------------
  extern "C" __global__ void __raygen__renderFrame()
  {
    // compute a test pattern based on pixel ID
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;


    const auto &camera = optixLaunchParams.camera;
    const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.fbSize.x;

    
    PRD prd;
    prd.random.init(ix+optixLaunchParams.frame.fbSize.x*iy,
                    optixLaunchParams.frame.frameID);
    prd.isPhoton = false;
    prd.pixelColor = vec3f(0.f);

    // the values we store the PRD pointer in:
    uint32_t u0, u1;
    packPointer( &prd, u0, u1 );

    int numPixelSamples = optixLaunchParams.numPixelSamples;


    vec3f pixelColor = 0.f;
    vec3f pixelNormal = 0.f;
    vec3f pixelAlbedo = 0.f;
    for (int sampleID=0;sampleID<numPixelSamples;sampleID++) {
      // normalized screen plane position, in [0,1]^2

      // iw: note for denoising that's not actually correct - if we
      // assume that the camera should only(!) cover the denoised
      // screen then the actual screen plane we shuld be using during
      // rendreing is slightly larger than [0,1]^2
      vec2f screen(vec2f(ix+prd.random(),iy+prd.random())
                   / vec2f(optixLaunchParams.frame.fbSize));
      // generate ray direction
      vec3f rayDir = normalize(camera.direction
                               + (screen.x - 0.5f) * camera.horizontal
                               + (screen.y - 0.5f) * camera.vertical);


      optixTrace(optixLaunchParams.traversable,
                 camera.position,
                 rayDir,
                 0.f,    // tmin
                 1e20f,  // tmax
                 0.0f,   // rayTime
                 OptixVisibilityMask( 255 ),
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
                 RADIANCE_RAY_TYPE,            // SBT offset
                 RAY_TYPE_COUNT,               // SBT stride
                 RADIANCE_RAY_TYPE,            // missSBTIndex 
                 u0, u1 );
       pixelColor  += prd.pixelColor;
       pixelNormal += prd.pixelNormal;
       pixelAlbedo += prd.pixelAlbedo;
    }
    // prd.position has the position on which the ray hit
    // loop the buffer and paint the pixels with the photons

    vec3f photonColor = vec3f(0.f, 0.f, 0.f);

    if (!optixLaunchParams.rayTrace) {
        pixelColor = vec3f(0.f, 0.f, 0.f);
    }

    if (optixLaunchParams.photonMap) {
        for (int photonID = 0; photonID < 1200 * optixLaunchParams.numOfPhotons * (optixLaunchParams.numOfBounces + 1); photonID++) {
            if (optixLaunchParams.photonArray[photonID].index > -1) {
                if (prd.position != vec3f(0.f, 0.f, 0.f)) {
                    vec3f diff = optixLaunchParams.photonArray[photonID].position - prd.position;
                    float xToPhoton = sqrtf(dot(diff, diff));
                    if (xToPhoton < optixLaunchParams.sphereRadius) {
                        float wpc = 1.f;// = (1.0 - xToPhoton / 10);
                        photonColor += optixLaunchParams.photonArray[photonID].color * wpc;
                    }
                }
            }
        }
    }
    
    pixelColor = pixelColor + photonColor / vec3f(M_PI * 4/3 * (optixLaunchParams.sphereRadius * optixLaunchParams.sphereRadius)) ;
    vec4f rgba(pixelColor / numPixelSamples, 1.f);
    // and write/accumulate to frame buffer ...
    if (optixLaunchParams.frame.frameID > 0) {
      rgba
        += float(optixLaunchParams.frame.frameID)
        *  vec4f(optixLaunchParams.frame.fbColor[fbIndex]);
      rgba /= (optixLaunchParams.frame.frameID+1.f);
    }
    optixLaunchParams.frame.fbColor[fbIndex] = (float4)rgba;
    optixLaunchParams.frame.fbFinal[fbIndex] = owl::make_rgba(rgba);
  }
  
} // ::osc

