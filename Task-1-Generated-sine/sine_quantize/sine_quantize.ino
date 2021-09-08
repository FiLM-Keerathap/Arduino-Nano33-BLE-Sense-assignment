/*

*/

// Import TensorFlow stuff
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Our model
#include "sine_model_quantize.h"

// Figure out what's going on in our model
#define DEBUG 1
const int kInferencesPerCycle = 1024; //จำนวน samples
const float Pi = 2.f * 3.14159265359f;// ค่า 2pi
// TFLite globals, used for compatibility with Arduino-style sketches
/*ตัวแปร globles ที่ให้tflite เข้าถึงได้*/
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  int i =0;
  
  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  /*จองพท.ให้ Tflite*/
  constexpr int kTensorArenaSize = 2 * 1024;//ประมาณ 2k
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

void setup() {
  #if DEBUG
  while(!Serial);
  #endif

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  /*ตั้งค่าให้ tflite รายงาน error ผ่าน serial ได้*/

  // Map the model into a usable data structure
  /*แปลงโมเดลเป็น structure ของบอรืด*/
  model = tflite::GetModel(sine_model_quantize);
  /*check ว่าบอร์ดใช้ได้ไหม*/
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  /*สร้างตัวล่าม*/
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
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


}

void loop() {
  unsigned long time_start,time_end;
  time_start = micros();
  float position = static_cast<float>(i) /
                   static_cast<float>(kInferencesPerCycle);
  // Copy value to input buffer (tensor)
  float x_val = Pi*position; /* ตัวแปร x เป็น 2Pi*(i/1024) โดยที่ i =0,1,2,...,1023 */
  
  /* Quantize the input from floating-point to integer
   แปลงจาก floating-point -> integer
  */
  int8_t x_quantized = x_val / model_input->params.scale + model_input->params.zero_point;
  model_input->data.int8[0] = x_quantized;/* ตอนนี้ model = sine( )*/

  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input: %f\n", x_val);
    return;
  }
  
  /* อ่าน ค่า y จาก buffer output ของ model*/ 
  int8_t y_quantized = model_output->data.int8[0]; 
  
  // Dequantize the output from integer to floating-point
  float y_val = (y_quantized - model_output->params.zero_point) * model_output->params.scale;

  // Print value
  Serial.println(10*y_val);/* คูณ gain*/
  //delay(50);
  i++; /*i นับไปเรื่อยๆ*/
  if (i >= kInferencesPerCycle) i = 0;/*กันค่า i overflow เมื่อ i >= 1024 ให้รีเซ็ทค่า i*/
  time_end = micros();
  Serial.print((float)(time_end - time_start)/1000);
  Serial.println("us");

}
