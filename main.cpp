// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

// #include <stdio.h>
// #include <fstream>
// #include <sstream>
// #include <vector>
// #include <algorithm>
// #include <k4a/k4a.h>
// #include <math.h>

// using namespace std;

// #ifdef __linux__
// int _stricmp(const char *a, const char *b) {
//   int ca, cb;
//   do {
//      ca = (unsigned char) *a++;
//      cb = (unsigned char) *b++;
//      ca = tolower(toupper(ca));
//      cb = tolower(toupper(cb));
//    } while (ca == cb && ca != '\0');
//    return ca - cb;
// }
// #endif

// // Enable HAVE_OPENCV macro after you installed opencv and opencv contrib modules (kinfu, viz), please refer to README.md
// #define HAVE_OPENCV
// #ifdef HAVE_OPENCV
// #include <opencv2/core.hpp>
// #include <opencv2/calib3d.hpp>
// #include <opencv2/opencv.hpp>
// #include <opencv2/rgbd.hpp>
// #include <opencv2/viz.hpp>
// using namespace cv;
// #endif

// #ifdef HAVE_OPENCV
// void initialize_kinfu_params(kinfu::Params &params,
//                              const int width,
//                              const int height,
//                              const float fx,
//                              const float fy,
//                              const float cx,
//                              const float cy)
// {
//     const Matx33f camera_matrix = Matx33f(fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
//     params.frameSize = Size(width, height);
//     params.intr = camera_matrix;
//     params.depthFactor = 1000.0f;
// }

// template<typename T> Mat create_mat_from_buffer(T *data, int width, int height, int channels = 1)
// {
//     Mat mat(height, width, CV_MAKETYPE(DataType<T>::type, channels));
//     memcpy(mat.data, data, width * height * channels * sizeof(T));
//     return mat;
// }
// #endif

// #define INVALID INT32_MIN
// typedef struct _pinhole_t
// {
//     float px;
//     float py;
//     float fx;
//     float fy;

//     int width;
//     int height;
// } pinhole_t;

// typedef struct _coordinate_t
// {
//     int x;
//     int y;
//     float weight[4];
// } coordinate_t;

// typedef enum
// {
//     INTERPOLATION_NEARESTNEIGHBOR, /**< Nearest neighbor interpolation */
//     INTERPOLATION_BILINEAR,        /**< Bilinear interpolation */
//     INTERPOLATION_BILINEAR_DEPTH   /**< Bilinear interpolation with invalidation when neighbor contain invalid
//                                                 data with value 0 */
// } interpolation_t;

// // Compute a conservative bounding box on the unit plane in which all the points have valid projections
// static void compute_xy_range(const k4a_calibration_t* calibration,
//     const k4a_calibration_type_t camera,
//     const int width,
//     const int height,
//     float& x_min,
//     float& x_max,
//     float& y_min,
//     float& y_max)
// {
//     // Step outward from the centre point until we find the bounds of valid projection
//     const float step_u = 0.25f;
//     const float step_v = 0.25f;
//     const float min_u = 0;
//     const float min_v = 0;
//     const float max_u = (float)width - 1;
//     const float max_v = (float)height - 1;
//     const float center_u = 0.5f * width;
//     const float center_v = 0.5f * height;

//     int valid;
//     k4a_float2_t p;
//     k4a_float3_t ray;

//     // search x_min
//     for (float uv[2] = { center_u, center_v }; uv[0] >= min_u; uv[0] -= step_u)
//     {
//         p.xy.x = uv[0];
//         p.xy.y = uv[1];
//         k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

//         if (!valid)
//         {
//             break;
//         }
//         x_min = ray.xyz.x;
//     }

//     // search x_max
//     for (float uv[2] = { center_u, center_v }; uv[0] <= max_u; uv[0] += step_u)
//     {
//         p.xy.x = uv[0];
//         p.xy.y = uv[1];
//         k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

//         if (!valid)
//         {
//             break;
//         }
//         x_max = ray.xyz.x;
//     }

//     // search y_min
//     for (float uv[2] = { center_u, center_v }; uv[1] >= min_v; uv[1] -= step_v)
//     {
//         p.xy.x = uv[0];
//         p.xy.y = uv[1];
//         k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

//         if (!valid)
//         {
//             break;
//         }
//         y_min = ray.xyz.y;
//     }

//     // search y_max
//     for (float uv[2] = { center_u, center_v }; uv[1] <= max_v; uv[1] += step_v)
//     {
//         p.xy.x = uv[0];
//         p.xy.y = uv[1];
//         k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

//         if (!valid)
//         {
//             break;
//         }
//         y_max = ray.xyz.y;
//     }
// }

// static pinhole_t create_pinhole_from_xy_range(const k4a_calibration_t* calibration, const k4a_calibration_type_t camera)
// {
//     int width = calibration->depth_camera_calibration.resolution_width;
//     int height = calibration->depth_camera_calibration.resolution_height;
//     if (camera == K4A_CALIBRATION_TYPE_COLOR)
//     {
//         width = calibration->color_camera_calibration.resolution_width;
//         height = calibration->color_camera_calibration.resolution_height;
//     }

//     float x_min = 0, x_max = 0, y_min = 0, y_max = 0;
//     compute_xy_range(calibration, camera, width, height, x_min, x_max, y_min, y_max);

//     pinhole_t pinhole;

//     float fx = 1.f / (x_max - x_min);
//     float fy = 1.f / (y_max - y_min);
//     float px = -x_min * fx;
//     float py = -y_min * fy;

//     pinhole.fx = fx * width;
//     pinhole.fy = fy * height;
//     pinhole.px = px * width;
//     pinhole.py = py * height;
//     pinhole.width = width;
//     pinhole.height = height;

//     return pinhole;
// }

// static void create_undistortion_lut(const k4a_calibration_t* calibration,
//     const k4a_calibration_type_t camera,
//     const pinhole_t* pinhole,
//     k4a_image_t lut,
//     interpolation_t type)
// {
//     coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

//     k4a_float3_t ray;
//     ray.xyz.z = 1.f;

//     int src_width = calibration->depth_camera_calibration.resolution_width;
//     int src_height = calibration->depth_camera_calibration.resolution_height;
//     if (camera == K4A_CALIBRATION_TYPE_COLOR)
//     {
//         src_width = calibration->color_camera_calibration.resolution_width;
//         src_height = calibration->color_camera_calibration.resolution_height;
//     }

//     for (int y = 0, idx = 0; y < pinhole->height; y++)
//     {
//         ray.xyz.y = ((float)y - pinhole->py) / pinhole->fy;

//         for (int x = 0; x < pinhole->width; x++, idx++)
//         {
//             ray.xyz.x = ((float)x - pinhole->px) / pinhole->fx;

//             k4a_float2_t distorted;
//             int valid;
//             k4a_calibration_3d_to_2d(calibration, &ray, camera, camera, &distorted, &valid);

//             coordinate_t src;
//             if (type == INTERPOLATION_NEARESTNEIGHBOR)
//             {
//                 // Remapping via nearest neighbor interpolation
//                 src.x = (int)floorf(distorted.xy.x + 0.5f);
//                 src.y = (int)floorf(distorted.xy.y + 0.5f);
//             }
//             else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
//             {
//                 // Remapping via bilinear interpolation
//                 src.x = (int)floorf(distorted.xy.x);
//                 src.y = (int)floorf(distorted.xy.y);
//             }
//             else
//             {
//                 printf("Unexpected interpolation type!\n");
//                 exit(-1);
//             }

//             if (valid && src.x >= 0 && src.x < src_width && src.y >= 0 && src.y < src_height)
//             {
//                 lut_data[idx] = src;

//                 if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
//                 {
//                     // Compute the floating point weights, using the distance from projected point src to the
//                     // image coordinate of the upper left neighbor
//                     float w_x = distorted.xy.x - src.x;
//                     float w_y = distorted.xy.y - src.y;
//                     float w0 = (1.f - w_x) * (1.f - w_y);
//                     float w1 = w_x * (1.f - w_y);
//                     float w2 = (1.f - w_x) * w_y;
//                     float w3 = w_x * w_y;

//                     // Fill into lut
//                     lut_data[idx].weight[0] = w0;
//                     lut_data[idx].weight[1] = w1;
//                     lut_data[idx].weight[2] = w2;
//                     lut_data[idx].weight[3] = w3;
//                 }
//             }
//             else
//             {
//                 lut_data[idx].x = INVALID;
//                 lut_data[idx].y = INVALID;
//             }
//         }
//     }
// }

// static void remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type)
// {
//     int src_width = k4a_image_get_width_pixels(src);
//     int dst_width = k4a_image_get_width_pixels(dst);
//     int dst_height = k4a_image_get_height_pixels(dst);

//     uint16_t* src_data = (uint16_t*)(void*)k4a_image_get_buffer(src);
//     uint16_t* dst_data = (uint16_t*)(void*)k4a_image_get_buffer(dst);
//     coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

//     memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(uint16_t));

//     for (int i = 0; i < dst_width * dst_height; i++)
//     {
//         if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
//         {
//             if (type == INTERPOLATION_NEARESTNEIGHBOR)
//             {
//                 dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
//             }
//             else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
//             {
//                 const uint16_t neighbors[4]{ src_data[lut_data[i].y * src_width + lut_data[i].x],
//                                              src_data[lut_data[i].y * src_width + lut_data[i].x + 1],
//                                              src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x],
//                                              src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x + 1] };

//                 // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
//                 // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
//                 // introduce noise on the edge. If the image is color or ir images, user should use
//                 // INTERPOLATION_BILINEAR
//                 if (type == INTERPOLATION_BILINEAR_DEPTH)
//                 {
//                     // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
//                     // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
//                     // introduce noise on the edge. If the image is color or ir images, user should use
//                     // INTERPOLATION_BILINEAR
//                     if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
//                     {
//                         continue;
//                     }

//                     // Ignore interpolation at large depth discontinuity without disrupting slanted surface
//                     // Skip interpolation threshold is estimated based on the following logic:
//                     // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
//                     // mode
//                     // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
//                     // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
//                     // cos(alpha)
//                     // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
//                     // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
//                     // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
//                     // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
//                     // smaller
//                     const float skip_interpolation_ratio = 0.04693441759f;
//                     float depth_min = min(min(neighbors[0], neighbors[1]), min(neighbors[2], neighbors[3]));
//                     float depth_max = max(max(neighbors[0], neighbors[1]), max(neighbors[2], neighbors[3]));
//                     float depth_delta = depth_max - depth_min;
//                     float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
//                     if (depth_delta > skip_interpolation_threshold)
//                     {
//                         continue;
//                     }
//                 }

//                 dst_data[i] = (uint16_t)(neighbors[0] * lut_data[i].weight[0] + neighbors[1] * lut_data[i].weight[1] +
//                     neighbors[2] * lut_data[i].weight[2] + neighbors[3] * lut_data[i].weight[3] +
//                     0.5f);
//             }
//             else
//             {
//                 printf("Unexpected interpolation type!\n");
//                 exit(-1);
//             }
//         }
//     }
// }

// void PrintUsage() 
// {
//     printf("Usage: kinfu_example.exe [Optional]<Mode>\n");
//     printf("    Mode: nfov_unbinned(default), wfov_2x2binned, wfov_unbinned, nfov_2x2binned\n");
//     printf("    Keys:   q - Quit\n");
//     printf("            r - Reset KinFu\n");
//     printf("            v - Enable Viz Render Cloud (default is OFF, enable it will slow down frame rate)\n");
//     printf("            w - Write out the kf_output.ply point cloud file in the running folder\n");
//     printf("    * Please ensure to uncomment HAVE_OPENCV pound define to enable the opencv code that runs kinfu\n");
//     printf("    * Please ensure to copy opencv/opencv_contrib/vtk dlls to the running folder\n\n");
// }

// int main(int argc, char** argv)
// {
//     PrintUsage();

//     k4a_device_t device = NULL;

//     if (argc > 2)
//     {
//         printf("Please read the Usage\n");
//         return 2;
//     }

//     // Configure the depth mode and fps
//     k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
//     config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
//     config.camera_fps = K4A_FRAMES_PER_SECOND_30;
//     if (argc == 2)
//     {
//         if (!_stricmp(argv[1], "nfov_unbinned"))
//         {
//             config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
//         }
//         else if (!_stricmp(argv[1], "wfov_2x2binned"))
//         {
//             config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
//         }
//         else if (!_stricmp(argv[1], "wfov_unbinned"))
//         {
//             config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
//             config.camera_fps = K4A_FRAMES_PER_SECOND_15;
//         }
//         else if (!_stricmp(argv[1], "nfov_2x2binned"))
//         {
//             config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
//         }
//         else if (!_stricmp(argv[1], "/?"))
//         {
//             return 0;
//         }
//         else
//         {
//             printf("Depth mode not supported!\n");
//             return 1;
//         }
//     }

//     uint32_t device_count = k4a_device_get_installed_count();

//     if (device_count == 0)
//     {
//         printf("No K4A devices found\n");
//         return 1;
//     }

//     if (K4A_RESULT_SUCCEEDED != k4a_device_open(K4A_DEVICE_DEFAULT, &device))
//     {
//         printf("Failed to open device\n");
//         k4a_device_close(device);
//         return 1;
//     }

//     // Retrive calibration
//     k4a_calibration_t calibration;
//     if (K4A_RESULT_SUCCEEDED !=
//         k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration))
//     {
//         printf("Failed to get calibration\n");
//         k4a_device_close(device);
//         return 1;
//     }

//     // Start cameras
//     if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(device, &config))
//     {
//         printf("Failed to start device\n");
//         k4a_device_close(device);
//         return 1;
//     }

//     // Generate a pinhole model for depth camera
//     pinhole_t pinhole = create_pinhole_from_xy_range(&calibration, K4A_CALIBRATION_TYPE_DEPTH);
//     interpolation_t interpolation_type = INTERPOLATION_BILINEAR_DEPTH;

// #ifdef HAVE_OPENCV
//     setUseOptimized(true);

//     // Retrieve calibration parameters
//     k4a_calibration_intrinsic_parameters_t *intrinsics = &calibration.depth_camera_calibration.intrinsics.parameters;
//     const int width = calibration.depth_camera_calibration.resolution_width;
//     const int height = calibration.depth_camera_calibration.resolution_height;

//     // Initialize kinfu parameters
//     Ptr<kinfu::Params> params;
//     params = kinfu::Params::defaultParams();
//     initialize_kinfu_params(
//         *params, width, height, pinhole.fx, pinhole.fy, pinhole.px, pinhole.py);

//     // Distortion coefficients
//     Matx<float, 1, 8> distCoeffs;
//     distCoeffs(0) = intrinsics->param.k1;
//     distCoeffs(1) = intrinsics->param.k2;
//     distCoeffs(2) = intrinsics->param.p1;
//     distCoeffs(3) = intrinsics->param.p2;
//     distCoeffs(4) = intrinsics->param.k3;
//     distCoeffs(5) = intrinsics->param.k4;
//     distCoeffs(6) = intrinsics->param.k5;
//     distCoeffs(7) = intrinsics->param.k6;

//     k4a_image_t lut = NULL;
//     k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
//                      pinhole.width,
//                      pinhole.height,
//                      pinhole.width * (int)sizeof(coordinate_t),
//                      &lut);

//     create_undistortion_lut(&calibration, K4A_CALIBRATION_TYPE_DEPTH, &pinhole, lut, interpolation_type);

//     // Create KinectFusion module instance
//     Ptr<kinfu::KinFu> kf;
//     kf = kinfu::KinFu::create(params);
//     namedWindow("AzureKinect KinectFusion Example");
//     viz::Viz3d visualization("AzureKinect KinectFusion Example");

//     bool stop = false;
//     bool renderViz = false;
//     k4a_capture_t capture = NULL;
//     k4a_image_t depth_image = NULL;
//     k4a_image_t undistorted_depth_image = NULL;
//     const int32_t TIMEOUT_IN_MS = 1000;
//     while (!stop && !visualization.wasStopped())
//     {
//         // Get a depth frame
//         switch (k4a_device_get_capture(device, &capture, TIMEOUT_IN_MS))
//         {
//         case K4A_WAIT_RESULT_SUCCEEDED:
//             break;
//         case K4A_WAIT_RESULT_TIMEOUT:
//             printf("Timed out waiting for a capture\n");
//             continue;
//             break;
//         case K4A_WAIT_RESULT_FAILED:
//             printf("Failed to read a capture\n");
//             k4a_device_close(device);
//             return 1;
//         }

//         // Retrieve depth image
//         depth_image = k4a_capture_get_depth_image(capture);
//         if (depth_image == NULL)
//         {
//             printf("Depth16 None\n");
//             k4a_capture_release(capture);
//             continue;
//         }

//         k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
//                          pinhole.width,
//                          pinhole.height,
//                          pinhole.width * (int)sizeof(uint16_t),
//                          &undistorted_depth_image);
//         remap(depth_image, lut, undistorted_depth_image, interpolation_type);

//         // Create frame from depth buffer
//         uint8_t *buffer = k4a_image_get_buffer(undistorted_depth_image);
//         uint16_t *depth_buffer = reinterpret_cast<uint16_t *>(buffer);
//         UMat undistortedFrame;
//         create_mat_from_buffer<uint16_t>(depth_buffer, width, height).copyTo(undistortedFrame);

//         if (undistortedFrame.empty())
//         {
//             k4a_image_release(depth_image);
//             k4a_image_release(undistorted_depth_image);
//             k4a_capture_release(capture);
//             continue;
//         }

//         // Update KinectFusion
//         if (!kf->update(undistortedFrame))
//         {
//             printf("Reset KinectFusion\n");
//             kf->reset();
//             k4a_image_release(depth_image);
//             k4a_image_release(undistorted_depth_image);
//             k4a_capture_release(capture);
//             continue;
//         }

//         // Retrieve rendered TSDF
//         UMat tsdfRender;
//         kf->render(tsdfRender);

//         // Retrieve fused point cloud and normals
//         UMat points;
//         UMat normals;
//         kf->getCloud(points, normals);

//         // Show TSDF rendering
//         imshow("AzureKinect KinectFusion Example", tsdfRender);

//         // Show fused point cloud and normals
//         if (!points.empty() && !normals.empty() && renderViz)
//         {
//             viz::WCloud cloud(points, viz::Color::white());
//             viz::WCloudNormals cloudNormals(points, normals, 1, 0.01, viz::Color::cyan());
//             visualization.showWidget("cloud", cloud);
//             visualization.showWidget("normals", cloudNormals);
//             visualization.showWidget("worldAxes", viz::WCoordinateSystem());
//             Vec3d volSize = kf->getParams().voxelSize * kf->getParams().volumeDims;
//             visualization.showWidget("cube", viz::WCube(Vec3d::all(0), volSize), kf->getParams().volumePose);
//             visualization.spinOnce(1, true);
//         }

//         // Key controls
//         const int32_t key = waitKey(5);
//         if (key == 'r')
//         {
//             printf("Reset KinectFusion\n");
//             kf->reset();
//         }
//         else if (key == 'v')
//         {
//             renderViz = true;
//         }
//         else if (key == 'w')
//         {
//             // Output the fused point cloud from KinectFusion
//             Mat out_points;
//             Mat out_normals;
//             points.copyTo(out_points);
//             normals.copyTo(out_normals);

//             printf("Saving fused point cloud into ply file ...\n");

//             // Save to the ply file
// #define PLY_START_HEADER "ply"
// #define PLY_END_HEADER "end_header"
// #define PLY_ASCII "format ascii 1.0"
// #define PLY_ELEMENT_VERTEX "element vertex"
//             string output_file_name = "kf_output.ply";
//             ofstream ofs(output_file_name); // text mode first
//             ofs << PLY_START_HEADER << endl;
//             ofs << PLY_ASCII << endl;
//             ofs << PLY_ELEMENT_VERTEX << " " << out_points.rows << endl;
//             ofs << "property float x" << endl;
//             ofs << "property float y" << endl;
//             ofs << "property float z" << endl;
//             ofs << "property float nx" << endl;
//             ofs << "property float ny" << endl;
//             ofs << "property float nz" << endl;
//             ofs << PLY_END_HEADER << endl;
//             ofs.close();

//             stringstream ss;
//             for (int i = 0; i < out_points.rows; ++i)
//             {
//                 ss << out_points.at<float>(i, 0) << " "
//                     << out_points.at<float>(i, 1) << " "
//                     << out_points.at<float>(i, 2) << " "
//                     << out_normals.at<float>(i, 0) << " "
//                     << out_normals.at<float>(i, 1) << " "
//                     << out_normals.at<float>(i, 2) << endl;
//             }
//             ofstream ofs_text(output_file_name, ios::out | ios::app);
//             ofs_text.write(ss.str().c_str(), (streamsize)ss.str().length());
//         }
//         else if (key == 'q')
//         {
//             stop = true;
//         }

//         k4a_image_release(depth_image);
//         k4a_image_release(undistorted_depth_image);
//         k4a_capture_release(capture);
//     }

//     k4a_image_release(lut);

//     destroyAllWindows();
// #endif

//     k4a_device_close(device);

//     return 0;
// }












// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <k4a/k4a.h>
#include <math.h>
#include"cnpy.h"

#include <chrono>
#include <thread>

#include <string>

//add the include for gitesh code
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues> 
#include <eigen3/unsupported/Eigen/SpecialFunctions>
#include "momentmatching.h"
#include <fstream>


using namespace std;

#ifdef __linux__
int _stricmp(const char *a, const char *b) {
  int ca, cb;
  do {
     ca = (unsigned char) *a++;
     cb = (unsigned char) *b++;
     ca = tolower(toupper(ca));
     cb = tolower(toupper(cb));
   } while (ca == cb && ca != '\0');
   return ca - cb;
}
#endif

// Enable HAVE_OPENCV macro after you installed opencv and opencv contrib modules (kinfu, viz), please refer to README.md
#define HAVE_OPENCV
#define HAVE_OPENCL
#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/viz.hpp>
using namespace cv;
#endif

#ifdef HAVE_OPENCV
void initialize_kinfu_params(kinfu::Params &params,
                             const int width,
                             const int height,
                             const float fx,
                             const float fy,
                             const float cx,
                             const float cy)
{
    const Matx33f camera_matrix = Matx33f(fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
    params.frameSize = Size(width, height);
    params.intr = camera_matrix;
    params.depthFactor = 1000.0f;
}

template<typename T> Mat create_mat_from_buffer(T *data, int width, int height, int channels = 1)
{
    Mat mat(height, width, CV_MAKETYPE(DataType<T>::type, channels));
    memcpy(mat.data, data, width * height * channels * sizeof(T));
    return mat;
}
#endif

#define INVALID INT32_MIN
typedef struct _pinhole_t
{
    float px;
    float py;
    float fx;
    float fy;

    int width;
    int height;
} pinhole_t;

typedef struct _coordinate_t
{
    int x;
    int y;
    float weight[4];
} coordinate_t;

typedef enum
{
    INTERPOLATION_NEARESTNEIGHBOR, /**< Nearest neighbor interpolation */
    INTERPOLATION_BILINEAR,        /**< Bilinear interpolation */
    INTERPOLATION_BILINEAR_DEPTH   /**< Bilinear interpolation with invalidation when neighbor contain invalid
                                                data with value 0 */
} interpolation_t;

// Compute a conservative bounding box on the unit plane in which all the points have valid projections
static void compute_xy_range(const k4a_calibration_t* calibration,
    const k4a_calibration_type_t camera,
    const int width,
    const int height,
    float& x_min,
    float& x_max,
    float& y_min,
    float& y_max)
{
    // Step outward from the centre point until we find the bounds of valid projection
    const float step_u = 0.25f;
    const float step_v = 0.25f;
    const float min_u = 0;
    const float min_v = 0;
    const float max_u = (float)width - 1;
    const float max_v = (float)height - 1;
    const float center_u = 0.5f * width;
    const float center_v = 0.5f * height;

    int valid;
    k4a_float2_t p;
    k4a_float3_t ray;

    // search x_min
    for (float uv[2] = { center_u, center_v }; uv[0] >= min_u; uv[0] -= step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_min = ray.xyz.x;
    }

    // search x_max
    for (float uv[2] = { center_u, center_v }; uv[0] <= max_u; uv[0] += step_u)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        x_max = ray.xyz.x;
    }

    // search y_min
    for (float uv[2] = { center_u, center_v }; uv[1] >= min_v; uv[1] -= step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_min = ray.xyz.y;
    }

    // search y_max
    for (float uv[2] = { center_u, center_v }; uv[1] <= max_v; uv[1] += step_v)
    {
        p.xy.x = uv[0];
        p.xy.y = uv[1];
        k4a_calibration_2d_to_3d(calibration, &p, 1.f, camera, camera, &ray, &valid);

        if (!valid)
        {
            break;
        }
        y_max = ray.xyz.y;
    }
}

static pinhole_t create_pinhole_from_xy_range(const k4a_calibration_t* calibration, const k4a_calibration_type_t camera)
{
    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        width = calibration->color_camera_calibration.resolution_width;
        height = calibration->color_camera_calibration.resolution_height;
    }

    float x_min = 0, x_max = 0, y_min = 0, y_max = 0;
    compute_xy_range(calibration, camera, width, height, x_min, x_max, y_min, y_max);

    pinhole_t pinhole;

    float fx = 1.f / (x_max - x_min);
    float fy = 1.f / (y_max - y_min);
    float px = -x_min * fx;
    float py = -y_min * fy;

    pinhole.fx = fx * width;
    pinhole.fy = fy * height;
    pinhole.px = px * width;
    pinhole.py = py * height;
    pinhole.width = width;
    pinhole.height = height;

    return pinhole;
}

static void create_undistortion_lut(const k4a_calibration_t* calibration,
    const k4a_calibration_type_t camera,
    const pinhole_t* pinhole,
    k4a_image_t lut,
    interpolation_t type)
{
    coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

    k4a_float3_t ray;
    ray.xyz.z = 1.f;

    int src_width = calibration->depth_camera_calibration.resolution_width;
    int src_height = calibration->depth_camera_calibration.resolution_height;
    if (camera == K4A_CALIBRATION_TYPE_COLOR)
    {
        src_width = calibration->color_camera_calibration.resolution_width;
        src_height = calibration->color_camera_calibration.resolution_height;
    }

    for (int y = 0, idx = 0; y < pinhole->height; y++)
    {
        ray.xyz.y = ((float)y - pinhole->py) / pinhole->fy;

        for (int x = 0; x < pinhole->width; x++, idx++)
        {
            ray.xyz.x = ((float)x - pinhole->px) / pinhole->fx;

            k4a_float2_t distorted;
            int valid;
            k4a_calibration_3d_to_2d(calibration, &ray, camera, camera, &distorted, &valid);

            coordinate_t src;
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                // Remapping via nearest neighbor interpolation
                src.x = (int)floorf(distorted.xy.x + 0.5f);
                src.y = (int)floorf(distorted.xy.y + 0.5f);
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                // Remapping via bilinear interpolation
                src.x = (int)floorf(distorted.xy.x);
                src.y = (int)floorf(distorted.xy.y);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }

            if (valid && src.x >= 0 && src.x < src_width && src.y >= 0 && src.y < src_height)
            {
                lut_data[idx] = src;

                if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // Compute the floating point weights, using the distance from projected point src to the
                    // image coordinate of the upper left neighbor
                    float w_x = distorted.xy.x - src.x;
                    float w_y = distorted.xy.y - src.y;
                    float w0 = (1.f - w_x) * (1.f - w_y);
                    float w1 = w_x * (1.f - w_y);
                    float w2 = (1.f - w_x) * w_y;
                    float w3 = w_x * w_y;

                    // Fill into lut
                    lut_data[idx].weight[0] = w0;
                    lut_data[idx].weight[1] = w1;
                    lut_data[idx].weight[2] = w2;
                    lut_data[idx].weight[3] = w3;
                }
            }
            else
            {
                lut_data[idx].x = INVALID;
                lut_data[idx].y = INVALID;
            }
        }
    }
}

static void remap(const k4a_image_t src, const k4a_image_t lut, k4a_image_t dst, interpolation_t type)
{
    int src_width = k4a_image_get_width_pixels(src);
    int dst_width = k4a_image_get_width_pixels(dst);
    int dst_height = k4a_image_get_height_pixels(dst);

    uint16_t* src_data = (uint16_t*)(void*)k4a_image_get_buffer(src);
    uint16_t* dst_data = (uint16_t*)(void*)k4a_image_get_buffer(dst);
    coordinate_t* lut_data = (coordinate_t*)(void*)k4a_image_get_buffer(lut);

    memset(dst_data, 0, (size_t)dst_width * (size_t)dst_height * sizeof(uint16_t));

    for (int i = 0; i < dst_width * dst_height; i++)
    {
        if (lut_data[i].x != INVALID && lut_data[i].y != INVALID)
        {
            if (type == INTERPOLATION_NEARESTNEIGHBOR)
            {
                dst_data[i] = src_data[lut_data[i].y * src_width + lut_data[i].x];
            }
            else if (type == INTERPOLATION_BILINEAR || type == INTERPOLATION_BILINEAR_DEPTH)
            {
                const uint16_t neighbors[4]{ src_data[lut_data[i].y * src_width + lut_data[i].x],
                                             src_data[lut_data[i].y * src_width + lut_data[i].x + 1],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x],
                                             src_data[(lut_data[i].y + 1) * src_width + lut_data[i].x + 1] };

                // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                // introduce noise on the edge. If the image is color or ir images, user should use
                // INTERPOLATION_BILINEAR
                if (type == INTERPOLATION_BILINEAR_DEPTH)
                {
                    // If the image contains invalid data, e.g. depth image contains value 0, ignore the bilinear
                    // interpolation for current target pixel if one of the neighbors contains invalid data to avoid
                    // introduce noise on the edge. If the image is color or ir images, user should use
                    // INTERPOLATION_BILINEAR
                    if (neighbors[0] == 0 || neighbors[1] == 0 || neighbors[2] == 0 || neighbors[3] == 0)
                    {
                        continue;
                    }

                    // Ignore interpolation at large depth discontinuity without disrupting slanted surface
                    // Skip interpolation threshold is estimated based on the following logic:
                    // - angle between two pixels is: theta = 0.234375 degree (120 degree / 512) in binning resolution
                    // mode
                    // - distance between two pixels at same depth approximately is: A ~= sin(theta) * depth
                    // - distance between two pixels at highly slanted surface (e.g. alpha = 85 degree) is: B = A /
                    // cos(alpha)
                    // - skip_interpolation_ratio ~= sin(theta) / cos(alpha)
                    // We use B as the threshold that to skip interpolation if the depth difference in the triangle is
                    // larger than B. This is a conservative threshold to estimate largest distance on a highly slanted
                    // surface at given depth, in reality, given distortion, distance, resolution difference, B can be
                    // smaller
                    const float skip_interpolation_ratio = 0.04693441759f;
                    float depth_min = min(min(neighbors[0], neighbors[1]), min(neighbors[2], neighbors[3]));
                    float depth_max = max(max(neighbors[0], neighbors[1]), max(neighbors[2], neighbors[3]));
                    float depth_delta = depth_max - depth_min;
                    float skip_interpolation_threshold = skip_interpolation_ratio * depth_min;
                    if (depth_delta > skip_interpolation_threshold)
                    {
                        continue;
                    }
                }

                dst_data[i] = (uint16_t)(neighbors[0] * lut_data[i].weight[0] + neighbors[1] * lut_data[i].weight[1] +
                    neighbors[2] * lut_data[i].weight[2] + neighbors[3] * lut_data[i].weight[3] +
                    0.5f);
            }
            else
            {
                printf("Unexpected interpolation type!\n");
                exit(-1);
            }
        }
    }
}

void PrintUsage() 
{
    printf("Usage: kinfu_example.exe [Optional]<Mode>\n");
    printf("    Mode: nfov_unbinned(default), wfov_2x2binned, wfov_unbinned, nfov_2x2binned\n");
    printf("    Keys:   q - Quit\n");
    printf("            r - Reset KinFu\n");
    printf("            v - Enable Viz Render Cloud (default is OFF, enable it will slow down frame rate)\n");
    printf("            w - Write out the kf_output.ply point cloud file in the running folder\n");
    printf("    * Please ensure to uncomment HAVE_OPENCV pound define to enable the opencv code that runs kinfu\n");
    printf("    * Please ensure to copy opencv/opencv_contrib/vtk dlls to the running folder\n\n");
}

 

int main(int argc, char** argv)
{   
    //------------------------------------------------modification
    using namespace Eigen;
    static constexpr int NUM_CLASSES = 7;
    Matrix<double, NUM_CLASSES, 2> input_dataSet;
    input_dataSet<< 1.,1/3.,5.,1/3.,7.,1/3.,8.,1/3.,2.,1/3.,4.,1/3.,6.,1/3.;
    // Matrix<double , NUM_CLASSES, 2>input_dataSet;
    // input_dataSet<< 
    //     //mu   sigma
    //     0.543, 0.065, //Concrete 
    //     0.577, 0.077, //Grass
    //     0.428, 0.059, //Pebbles
    //     0.478, 0.113, //Rocks
    //     0.372, 0.055, //Wood
    //     0.616, 0.048, //Rubber
    //     0.583, 0.068; //Rug
    // Matrix<double , NUM_CLASSES, 2>input_dataSet;
    // input_dataSet<< 
    //     //mu   sigma
    //     5.43, 0.065, //Concrete 
    //     5.77, 0.077, //Grass
    //     4.28, 0.059, //Pebbles
    //     4.78, 0.113, //Rocks
    //     3.72, 0.055, //Wood
    //     6.16, 0.048, //Rubber
    //     5.83, 0.068; //Rug


    // Matrix<double , NUM_CLASSES, 1> input_a;
    // input_a << 0.,1.,0.,2.,0.,8.,0.;
    Matrix<double , 7, 1> input_a ;
    input_a << 2., 14., 2., 10.,4., 11., 3.;
    // vector<double> measurements = {4.99164777 ,4.77152141 ,3.91641394, 4.55800433 ,3.8606777 , 3.74498796,
    // 5.72879082 ,5.7962263 , 5.07160451 ,4.90461636};
    // vector<double> measurements = {0.62,0.62,0.62,0.62,0.62,0.62,0.62,0.62,0.62,0.62};
    vector<double> measurements = {6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2};
    momentmatching(input_dataSet, input_a, measurements);
    //------------------------------------------------modification

    
    PrintUsage();

    k4a_device_t device = NULL;

    if (argc > 2)
    {
        printf("Please read the Usage\n");
        return 2;
    }

    // Configure the depth mode and fps
    k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    if (argc == 2)
    {
        if (!_stricmp(argv[1], "nfov_unbinned"))
        {
            config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        }
        else if (!_stricmp(argv[1], "wfov_2x2binned"))
        {
            config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
        }
        else if (!_stricmp(argv[1], "wfov_unbinned"))
        {
            config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
            config.camera_fps = K4A_FRAMES_PER_SECOND_15;
        }
        else if (!_stricmp(argv[1], "nfov_2x2binned"))
        {
            config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
        }
        else if (!_stricmp(argv[1], "/?"))
        {
            return 0;
        }
        else
        {
            printf("Depth mode not supported!\n");
            return 1;
        }
    }

    // uint32_t device_count = k4a_device_get_installed_count();

    // if (device_count == 0)
    // {
    //     printf("No K4A devices found\n");
    //     return 1;
    // }

    // if (K4A_RESULT_SUCCEEDED != k4a_device_open(K4A_DEVICE_DEFAULT, &device))
    // {
    //     printf("Failed to open device\n");
    //     k4a_device_close(device);
    //     return 1;
    // }

    // // Retrive calibration
    // k4a_calibration_t calibration;
    // if (K4A_RESULT_SUCCEEDED !=
    //     k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration))
    // {
    //     printf("Failed to get calibration\n");
    //     k4a_device_close(device);
    //     return 1;
    // }

    // // Start cameras
    // if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(device, &config))
    // {
    //     printf("Failed to start device\n");
    //     k4a_device_close(device);
    //     return 1;
    // }

    // //Generate a pinhole model for depth camera
    // pinhole_t pinhole = create_pinhole_from_xy_range(&calibration, K4A_CALIBRATION_TYPE_DEPTH);
    // interpolation_t interpolation_type = INTERPOLATION_BILINEAR_DEPTH;

#ifdef HAVE_OPENCV
    // setUseOptimized(true);
    // Disable Multithreading processing
    setNumThreads(0);

//     // Retrieve calibration parameters
//     k4a_calibration_intrinsic_parameters_t *intrinsics = &calibration.depth_camera_calibration.intrinsics.parameters;
    const int width = 1280; //calibration.depth_camera_calibration.resolution_width;
    const int height = 720; //calibration.depth_camera_calibration.resolution_height;

    // Initialize kinfu parameters
    Ptr<kinfu::Params> params;
    params = kinfu::Params::defaultParams();
    initialize_kinfu_params(*params, width, height, 525, 525, width/2-0.5f, height/2-0.5f);
    //initialize_kinfu_params(*params, width, height, pinhole.fx, pinhole.fy, pinhole.px, pinhole.py);

    // // Distortion coefficients
    // Matx<float, 1, 8> distCoeffs;
    // distCoeffs(0) = intrinsics->param.k1;
    // distCoeffs(1) = intrinsics->param.k2;
    // distCoeffs(2) = intrinsics->param.p1;
    // distCoeffs(3) = intrinsics->param.p2;
    // distCoeffs(4) = intrinsics->param.k3;
    // distCoeffs(5) = intrinsics->param.k4;
    // distCoeffs(6) = intrinsics->param.k5;
    // distCoeffs(7) = intrinsics->param.k6;

    // k4a_image_t lut = NULL;
    // k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
    //                  pinhole.width,
    //                  pinhole.height,
    //                  pinhole.width * (int)sizeof(coordinate_t),
    //                  &lut);

    // create_undistortion_lut(&calibration, K4A_CALIBRATION_TYPE_DEPTH, &pinhole, lut, interpolation_type);

    // Create KinectFusion module instance
    Ptr<kinfu::KinFu> kf;
    kf = kinfu::KinFu::create(params);
    namedWindow("AzureKinect KinectFusion Example");
    viz::Viz3d visualization("AzureKinect KinectFusion Example");

    bool stop = false;
    bool renderViz = false;
    k4a_capture_t capture = NULL;
    k4a_image_t depth_image = NULL;
    k4a_image_t undistorted_depth_image = NULL;
    const int32_t TIMEOUT_IN_MS = 1000;
    int i = 0;



    //----------------------------------------------------------------------modification 
    // static constexpr int NUM_CLASSES = 7;
    //define the measurement vector 
    // vector<double> measurements;
    Matrix<double, NUM_CLASSES, 2> dataSet;
    dataSet<< 1.,1/3.,5.,1/3.,7.,1/3.,8.,1/3.,2.,1/3.,4.,1/3.,6.,1/3.;
    bool stope_video_input = false;
    bool test = false;
    //----------------------------------------------------------------------modification 


    while (i <= 299 && !stop && !visualization.wasStopped())
    {
        printf("%d\n", i);

        // Get a depth frame
        //UNCOMMENT THIS CODE IF YOU WANT TO USE LIVE DATA
        // switch (k4a_device_get_capture(device, &capture, TIMEOUT_IN_MS))
        // {
        // case K4A_WAIT_RESULT_SUCCEEDED:
        //     break;
        // case K4A_WAIT_RESULT_TIMEOUT:
        //     printf("Timed out waiting for a capture\n");
        //     continue;
        //     break;
        // case K4A_WAIT_RESULT_FAILED:
        //     printf("Failed to read a capture\n");
        //     k4a_device_close(device);
        //     return 1;
        // }

        // // Retrieve depth image
        // depth_image = k4a_capture_get_depth_image(capture);
        // if (depth_image == NULL)
        // {
        //     printf("Depth16 None\n");
        //     k4a_capture_release(capture);
        //     continue;
        // }

        // k4a_image_create(K4A_IMAGE_FORMAT_DEPTH16,
        //                  pinhole.width,
        //                  pinhole.height,
        //                  pinhole.width * (int)sizeof(uint16_t),
        //                  &undistorted_depth_image);
        // remap(depth_image, lut, undistorted_depth_image, interpolation_type);

        // // Create frame from depth buffer
        // uint8_t *buffer = k4a_image_get_buffer(undistorted_depth_image);



        //LOAD IN DEPTH IMAGE



        std::string str;
        // std::string s1 = "For_KINFU_mapping_without_tape_out/";
        std::string s1 = "red/";

        std::string s2 = std::to_string(i);
        std::string s3 = "_depth.npy";
        str.append(s1);
        str.append(s2);
        str.append(s3);
        cnpy::NpyArray arr = cnpy::npy_load(str);
        double* loaded_data = arr.data<double>();
        

        int dim_1 = arr.shape[0];
        int dim_2 = arr.shape[1];

        int short_dims[] = {dim_1, dim_2};

        // std::string s = "originals/1_depth.npy";
        // cnpy::NpyArray array = cnpy::npy_load(s);
        // double* ptr = array.data<double>();
        // cv::Mat m = cv::Mat(2, short_dims, CV_64F, ptr).clone();
        // cv::imshow("Output", m);

        // CONVERT DEPTH IMAGE TO OPENCV MATRIX
        cv::Mat depth = cv::Mat(2, short_dims,  CV_16S, loaded_data).clone();

      
        // GET THE SEMANTIC INFORMATION
        std::string s4 = "_semantic.npy";
        std::string s;
        s.append(s1);
        s.append(s2);
        s.append(s4);
        cnpy::NpyArray semantic_arr = cnpy::npy_load(s);
        uint8_t* ptr = semantic_arr.data<uint8_t>();
        // CONVERT SEMANTIC INFORMATION TO OPENCV MATRIX
        cv::Mat semantic = cv::Mat(2, short_dims, CV_8U, ptr).clone();
        //imshow("semantic_mat", semantic);

        
        //IN CASE YOU WANT TO USE THE ORIGINAL FILES-- THIS IS OLD WORK, NOT SURE IF IT WORKS ANYMORE
        // cv::imshow("out", oned);
        // cv::Mat mnd = cv::Mat(3, dims, CV_64F, loaded_data).clone();
        // cv::Mat m2= Mat(height, width, CV_32F, 0.0);
        // // cv::Mat mnd(3, dims, CV_64F);
        // // extract planes from matrix
        // std::vector<cv::Mat> matVec;
        // // Mat mtest = Mat(760, 1280, CV_8U);
        // for (int p = 0; p < dims[2]; ++p) {
        //     double *ind = (double*)mnd.data + p * dims[0] * dims[1]; // sub-matrix pointer
        //     matVec.push_back(cv::Mat(2, short_dims, CV_64F, ind).clone()); // clone if mnd goes away
        // }
        // std::cout << "Size of matVec: " << matVec.size() << std::endl;
        // std::cout << "Size of first Mat: " << matVec[0].size() << std::endl;
        //cv::imshow("Output", matVec[0]);
        // std::cout << "\nmatVec[0]:\n" << matVec[0] << std::endl;
        // std::cout << "\nmatVec[1]:\n" << matVec[1] < std::endl;
        // std::cout << "\nmatVec[2]:\n" << matVec[2] << std::endl;

        
        //int dims[3] = {dim_1, dim_2, dim_3};
        //int size_one[2] = {dim_1, dim_2};
        
        //cv::imwrite("direct_frame.jpg", m1);
        // std::vector<cv::Mat> matVec;
        // for (int p = 0; p < dims[2]; ++p){
        //     float *ind = (float*)m1.data + p*dims[0]*dims[1];
        //     matVec.push_back(cv::Mat(2, dims, CV_32F, ind).clone());
        //     if (p == 0){
        //         cv::imwrite("first_channel_frame.jpg", matVec[0]);
        //     }
        // }
        //cv::split(m1, bands);
        // cv::Mat m3 = cv::Mat(2, size_one, CV_32F);
        // for (int a = 0; a < 720; ++a){
        //     for (int b = 0; b < 1280; ++b){
        //         float
        //         m3.at<float>(a, b) = bands[0].at<float>(a, b);
        //     }
        // }
        //  cv::imshow("frame.jpg", m3);

        // printf("%d %d ", m1.rows, m1.channels());
        // int width = 1280;
        // int height = 720;
        // printf("%d %d ", m1.dims, m2.dims);
        // for (int a = 0; a < height; ++a){
        //     for (int b = 0; b < width; ++b){
        //         if (m1.at<float>(a, b, 0) != 0.0){
        //             printf("%f ", m1.at<float>(a, b, 1));
                    
        //         }
        //         m2.at<float>(a, b) = m1.at<float>(a, b, 1);
        //         // printf("%f\n", m1.at<float>(a, b, 1));
        //         //m1.at<uchar>(a, b) = loaded_data[width*58*a + 58*b];
        //         int max_class = 0;
        //         float max_prob = 0.0;
        //         for (int c = 1; c < 58; ++c){
        //             if (m1.at<float>(a, b, c) > max_prob){
        //                 max_prob = m1.at<float>(a, b, c);
        //                 max_class = c;
        //             }
        //         }
        //     }
        // }
        // //cv::imwrite("frame.jpg", m2);

        //CONVERT TO UMAT
        UMat undistortedFrame = depth.getUMat(ACCESS_READ);
        // printf("made it here\n");


        // for (int i = 0; i < arr.shape.size(); ++i){
        //     printf("%u ", arr.shape[i]);
        // }
        if (undistortedFrame.empty())
        {
            printf("Frame was empty");
            //k4a_image_release(depth_image);
            //k4a_image_release(undistorted_depth_image);
            //k4a_capture_release(capture);
            continue;
        }


        // Update KinectFusion
        // printf("and here too\n");
        if (!kf->update(depth, semantic,stope_video_input))
        {
            printf("Reset KinectFusion\n");
            kf->reset();
            //k4a_image_release(depth_image);
            //k4a_image_release(undistorted_depth_image);
            //k4a_capture_release(capture);
            if (test == true){
                stope_video_input = true;
            }
            continue;
        }
        std::cout << "-----------------------------------seg-------------------1" << std::endl;
        // Retrieve rendered TSDF
        UMat tsdfRender;
        kf->render(tsdfRender);
        std::cout << "-----------------------------------seg-------------------4" << std::endl;
        // Retrieve fused point cloud and normals
        UMat points;
        UMat normals;
        //printf("also here\n");
        //kf->getCloud(points, normals);



        //Show TSDF rendering
        imshow("AzureKinect KinectFusion Example", tsdfRender);



        //-------------------------------------------------------------------------modification 
        // std::string image_direction = "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/For_KINFU_mapping_without_tape_out/kinfu_only_depth/";
        // std::string image_name = std::to_string(i)+".jpg";
        // std::string name = image_direction+image_name;
        // std::cout << name << std::endl;
        // cv::imwrite( name,tsdfRender);
        //-------------------------------------------------------------------------modification 






        // Show fused point cloud and normals
        //if (!points.empty() && !normals.empty() && renderViz)
        //{
            //viz::WCloud cloud(points, viz::Color::white());
            //viz::WCloudNormals cloudNormals(points, normals, 1, 0.01, viz::Color::cyan());
            //visualization.showWidget("cloud", cloud);
            //visualization.showWidget("normals", cloudNormals);
            //visualization.showWidget("worldAxes", viz::WCoordinateSystem());
            //Vec3d volSize = kf->getParams().voxelSize * kf->getParams().volumeDims;
            //visualization.showWidget("cube", viz::WCube(Vec3d::all(0), volSize), kf->getParams().volumePose);
            //visualization.spinOnce(1, true);
        //}
        // printf("also here\n");
        // Key controls
        //define the class_index_number 
         


        const int32_t key = waitKey(5);
        if (key == 'r')
        {
            printf("Reset KinectFusion\n");
            kf->reset();
        }
        else if (key == 'v')
        {
            renderViz = true;
        }
        else if (key == 'w')
        {
            // Output the fused point cloud from KinectFusion
            Mat out_points;
            Mat out_normals;
            points.copyTo(out_points);
            normals.copyTo(out_normals);

            printf("Saving fused point cloud into ply file ...\n");

            // Save to the ply file
#define PLY_START_HEADER "ply"
#define PLY_END_HEADER "end_header"
#define PLY_ASCII "format ascii 1.0"
#define PLY_ELEMENT_VERTEX "element vertex"
            string output_file_name = "kf_output.ply";
            ofstream ofs(output_file_name); // text mode first
            ofs << PLY_START_HEADER << endl;
            ofs << PLY_ASCII << endl;
            ofs << PLY_ELEMENT_VERTEX << " " << out_points.rows << endl;
            ofs << "property float x" << endl;
            ofs << "property float y" << endl;
            ofs << "property float z" << endl;
            ofs << "property float nx" << endl;
            ofs << "property float ny" << endl;
            ofs << "property float nz" << endl;
            ofs << PLY_END_HEADER << endl;
            ofs.close();

            stringstream ss;
            for (int i = 0; i < out_points.rows; ++i)
            {
                ss << out_points.at<float>(i, 0) << " "
                    << out_points.at<float>(i, 1) << " "
                    << out_points.at<float>(i, 2) << " "
                    << out_normals.at<float>(i, 0) << " "
                    << out_normals.at<float>(i, 1) << " "
                    << out_normals.at<float>(i, 2) << endl;
            }
            ofstream ofs_text(output_file_name, ios::out | ios::app);
            ofs_text.write(ss.str().c_str(), (streamsize)ss.str().length());
        }
        else if (key == 'q')
        {
            stop = true;
        }
        else if (key == 'c'){
            // //define the class_index_number 
            // static constexpr int NUM_CLASSES = 7;
            // //define the measurement vector 
            // vector<double> measurements;
            // Matrix<double, NUM_CLASSES, 2> dataSet;
            // dataSet<< 1.,1/3.,5.,1/3.,7.,1/3.,8.,1/3.,2.,1/3.,4.,1/3.,6.,1/3.;

            //read the data from the txt file
            std::ifstream file ("measurement_data.txt");
            if (file.is_open() == false){
                std::cout << "the measurement_data file can't be open" << std::endl;
            }
            std::string line;
            while(std::getline(file, line)){
                double data = std::stod(line);
                measurements.push_back(data);
            }
            std::cout<< measurements.size() << std::endl;


            
            int class_index;
            int x, y;
            std::cout << "please cin the x coordinate and y coordinate" << std::endl;
            cin >> x >> y;
            std::cout << "please cin the class_index" << std::endl;
            cin >> class_index;

            kf->update_friction(std::make_pair(x,y),class_index,&dataSet,measurements);
            std::cout << dataSet(1,0) << std::endl;

            //we stope the input video 
            test = true;
        }
        std::cout << "the new value is ---------->" << dataSet(1,0) << std::endl;
        //k4a_image_release(depth_image);
        // printf("plus here\n");
        // printf("Beginning 2");
        // //k4a_image_release(undistorted_depth_image);
        // printf("End 2");
        // printf("and finally here\n");
        //k4a_capture_release(capture);
        ++i;
    }

    //k4a_image_release(lut);

    destroyAllWindows();
#endif

    // k4a_device_close(device);


    return 0;
}