# Arduino-Nano33-BLE-Sense-assignment

<p> ลองเล่น TFLite กับบอร์ด Arduino Nano 33 BLE </p>
<h4><b>Required:</b> Colab, Arduino Nano 33 BLE Sense, Arduino IDE</h4>

- [Getting started with the Arduino Nano 33 BLE Sense](https://www.arduino.cc/en/Guide/NANO33BLESense#use-your-arduino-nano-33-ble-sense-on-the-arduino-web-ide)<p>
for setup your Arduino Nano 33 BLE Sense for the first time. 
</p>

- (Optional) [Netron](https://github.com/lutzroeder/netron) <p>
for visualizing your  neural network, deep learning and machine learning models
</p>

## [Task-1-Generated-sine](Task-1-Generated-sine)

<p>
 เขียนโมเดลสร้างสัญญาณไซน์ โดยอินพุตจะเป็นค่า [ 0.0, 2π ] และมีเอาต์พุตที่เป็นไปได้คือ [ -1.0, 1.0 ]
</p>

1. sine <p> ใช้ Converter แบบ DEFAULT จะเป็นการแปลงรุ่นแบบเริ่มต้น โมเดลจะทำงานด้วยพารามิเตอร์ float32</p>
2. sine_quantize <p> ใช้ CONVERTER แบบ Post-training integer quantization ทำการแปลงรุ่นโมเดลให้ทำงานด้วยตัวแปร int_8</p>


## [Task--2-Anomaly-detection](Task-2-Autoencoder)
<p>
 เขียนโมเดลสำหรับตรวจจับความผิดปกติของข้อมูล โดยใช้ชุดข้อมูลจากโฟล์เดอร์ [data](data)</p> [data](data)
