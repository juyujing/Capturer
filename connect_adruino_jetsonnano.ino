/******************************************************************************************************
 * 单片机：mega328p-au 外部晶振：16M 
********************************************************************************************/
#include <SoftwareSerial.h>    //包含软串口头文件，硬串口通信文件库系统自带   
String uart1_receive_buf = "";
char cmd_return[101];
char str;
/*******************************一些宏定义****************************************/
SoftwareSerial mySerial(A4,A5); //创建一个软串口的类，模拟引脚4,5分别代表 RX, TX引脚 AR多功能板
//SoftwareSerial mySerial(10,2);  //创建一个软串口的类，模拟引脚4,5分别代表 RX, TX引脚 AR扩展版
    
void setup() {
    Serial.begin(115200);           //硬件串口
    mySerial.begin(115200);         //设置软串口波特率
}

void loop() {
  if ( Serial.available())
    {
      str=Serial.read();
      if('f' == str){
          Serial.println("forward");    
          mySerial.print("$DGT:52-55,1!");  //总线口 调用 52到55 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('l' == str){
          Serial.println("left");    
          mySerial.print("$DGT:60-63,1!");  //总线口 调用 60到63 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('r' == str){
          Serial.println("right");    
          mySerial.print("$DGT:64-67,1!");  //总线口 调用 64到67 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('h' == str){
          Serial.println("catch");    
          mySerial.print("$DGT:402-410,1!");  //总线口 调用 398到406 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('d' == str){
          Serial.println("down");    
          mySerial.print("$DGS:439!");  //总线口 调用 437 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('b' == str){
          Serial.println("back");    
          mySerial.print("$DGT:56-59,1!");  //总线口 调用 56到59 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('q' == str){
          Serial.println("slightly down");    
          mySerial.print("$DGS:438!");  //总线口 调用 436 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('z' == str){
          Serial.println("slightly left");    
          mySerial.print("$DGT:441-444,1!");  //总线口 调用 439到442 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('y' == str){
          Serial.println("slightly right");    
          mySerial.print("$DGT:445-448,1!");  //总线口 调用 459到462 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('t' == str){
          Serial.println("four leg back");    
          mySerial.print("$DGT:431-434,1!");  //总线口 调用 431到434 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('a' == str){
          Serial.println("four leg left");    
          mySerial.print("$DGT:425-428,1!");  //总线口 调用 421到424 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('m' == str){
          Serial.println("four leg forward");    
          mySerial.print("$DGT:419-422,1!");  //总线口 调用 421到424 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
      if('p' == str){
          Serial.println("reset");    
          mySerial.print("$DGS:2!");  //总线口 调用 421到424 动作，执行1次 其他命令参照控制器指令
          delay(2);     
      }
}
}






