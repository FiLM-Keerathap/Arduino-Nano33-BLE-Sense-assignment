/*
Input: PSD of vibration signal [array size 128]
Model = autoencoder
Output: PSD hat [array size 128]
approximation MAE and use it for decision making
colab:https://colab.research.google.com/drive/19WM3A7vwDPaFMtaEg_EG2ly-YR-VUBCx?usp=sharing (last update 22/9/2564 22.46)
*/

/*
ERROR
Didn't find op for builtin opcode 'TRANSPOSE_CONV' version '3'. 
An older version of this builtin might be supported. 
Are you using an old TFLite binary with a newer model?

Failed to get registration from op code TRANSPOSE_CONV
*/
// Import TensorFlow stuff

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Our model
#include "Conv2D-autoencoder_model.h"
#include "anomaly_sample.h"
#include "normal_sample.h"
// Testing
#include "tensorflow/lite/micro/testing/micro_test.h"
//helper function
#include "utils.h"

#define DEBUG 1

const int MAX_MEASUREMENTS = 128; //จำนวน samples
//const int dim = 1; //จำนวน dimension
float threshold = 1.2373894742152644e-05;

// TFLite globals, used for compatibility with Arduino-style sketches
/*ตัวแปร globles ที่ให้tflite เข้าถึงได้*/
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  
  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  /*จองพท.ให้ Tflite*/
  const int tensor_arena_size = 20*1024;//ประมาณ 20k
  uint8_t tensor_arena[tensor_arena_size];
} // namespace
/************************************************************************************************************************************/
void setup() {
  #if DEBUG
  Serial.begin(115200);
  while (!Serial);
  Serial.println("TensorFlow Lite test");
  Serial.println();
  #endif

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  /*ตั้งค่าให้ tflite รายงาน error ผ่าน serial ได้*/

  // Map the model into a usable data structure
  /*แปลงโมเดลเป็น structure ของบอรืด*/
  model = tflite::GetModel(models_Conv2D_autoencoder_model_tflite);
  /*check ว่าบอร์ดใช้ได้ไหม*/
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    //error_reporter->Report("Model version does not match Schema");
    TF_LITE_REPORT_ERROR(error_reporter,
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
    while(1);
  }

  static tflite::AllOpsResolver resolver;
  /*static tflite::MicroMutableOpResolver<1> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }*/

  // Build an interpreter to run the model
  /*สร้างตัวล่าม*/
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, tensor_arena_size, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }
  /*ต่อ ตัวแปรเข้ากับ buffers (แปลงเป็น tensor)*/
  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Dim 3 size: ");
  Serial.println(model_input->dims->data[2]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
}

void loop() {
  
  float measurements[MAX_MEASUREMENTS];
  float y_predict[MAX_MEASUREMENTS];
  float MAE ;
  int8_t x_quantized,y_quantized;
  unsigned long time_start,time_end;
  
  /**********************************************************************************************************************/
  Serial.println("*********************************************");
  Serial.println("---Normal Sample---");
  time_start = micros();
  for (int i = 0; i < MAX_MEASUREMENTS; i++) {
    measurements[i] = normal_sample[i];
    x_quantized = (normal_sample[i] / model_input->params.scale) + model_input->params.zero_point;
    model_input->data.int8[i] = x_quantized;
  }
  
  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
    return;
  }

  // Read predicted y value from output buffer (tensor)
  for (int i = 0; i < MAX_MEASUREMENTS; i++) {
    y_quantized = model_output->data.int8[i];
    // Dequantize the output from integer to floating-point
    y_predict[i] = (y_quantized - model_output->params.zero_point) * model_output->params.scale;
  }
  time_end = micros();
  Serial.println("RUN MODEL!!");
  Serial.print("latency :\t");
  Serial.print((float)(time_end - time_start));
  Serial.println("\tus");
  /*******************************************************************************************************************/
  time_start = micros();
  //calculate MAE
  MAE = calc_mae(measurements, y_predict, MAX_MEASUREMENTS);
  Serial.print("MAE :\t");
  Serial.println(MAE,8);
  time_end = micros();
  Serial.print("latency :\t");
  Serial.print((float)(time_end - time_start));
  Serial.println("\tus");
  /*******************************************************************************************************************/
  time_start = micros();
  //Decision making
  if(isgreater(MAE, threshold))
     Serial.println("Predict: Abnormal");
  else
     Serial.println("Predict: Normal");
  time_end = micros();
  Serial.print("latency :\t");
  Serial.print((float)(time_end - time_start));
  Serial.println("\tus");
  /**********************************************************************************************************************/
  Serial.println("*********************************************");
  Serial.println("---Anormaly Sample---");
  time_start = micros();
  for (int i = 0; i < MAX_MEASUREMENTS; i++) {
      measurements[i] = anomaly_sample[i];
      x_quantized = anomaly_sample[i] / model_input->params.scale + model_input->params.zero_point;
      model_input->data.int8[i] = x_quantized;
  }
  
  // Read predicted y value from output buffer (tensor)
  for (int i = 0; i < MAX_MEASUREMENTS; i++) {
    y_quantized = model_output->data.int8[i];
    // Dequantize the output from integer to floating-point
    y_predict[i] = (y_quantized - model_output->params.zero_point) * model_output->params.scale;
  }
  time_end = micros();
  Serial.println("RUN MODEL!!");
  Serial.print("latency :\t");
  Serial.print((float)(time_end - time_start));
  Serial.println("\tus");
  /*******************************************************************************************************************/
  time_start = micros();
  //calculate MAE
  MAE = calc_mae(measurements, y_predict, MAX_MEASUREMENTS);
  Serial.print("MAE :\t");
  Serial.println(MAE,8);
  time_end = micros();
  Serial.print("latency :\t");
  Serial.print((float)(time_end - time_start));
  Serial.println("\tus");
  /*******************************************************************************************************************/
  time_start = micros();
  //Decision making
  if(isgreater(MAE, threshold))
     Serial.println("Predict: Abnormal");
  else
     Serial.println("Predict: Normal");
  time_end = micros();
  Serial.print("latency :\t");
  Serial.print((float)(time_end - time_start));
  Serial.println("\tus");
  while(1);
}
