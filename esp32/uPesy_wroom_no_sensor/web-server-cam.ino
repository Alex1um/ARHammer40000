 
#include <WiFi.h>
#include <WebServer.h>

#define ESP_MODEL_WROOM
#if defined(CAMERA_MODEL_AI_THINKER)
  #define MOTOR_1_PIN_1    14
  #define MOTOR_1_PIN_2    15
  #define MOTOR_2_PIN_1    13
  #define MOTOR_2_PIN_2    12
  #define LED_PIN 33
  #define LED_HIGH LOW
  #define LED_LOW HIGH
  #define MOTOR_SPEED_PIN_1 2
  #define MOTOR_SPEED_PIN_2 4
#elif defined(ESP_MODEL_WROOM)
  #define MOTOR_1_PIN_1    32
  #define MOTOR_1_PIN_2    33
  #define MOTOR_2_PIN_1    25
  #define MOTOR_2_PIN_2    26
  #define LED_PIN 2
  #define LED_HIGH HIGH
  #define LED_LOW LOW
  #define SDA_PIN 21
  #define SCL_PIN 22
  #define MOTOR_SPEED_PIN_1 13
  #define MOTOR_SPEED_PIN_2 14
#endif

WebServer server(80);

const char *ssid = "Hackspace";
// const char *pass = "you@hackspace";
const char *pass = "youathackspace";
// const char *ssid = "just another spot";
// const char *pass = "11111111";

uint8_t SPEED = 255;

void go_left() {
  digitalWrite(MOTOR_1_PIN_1, 1);
  digitalWrite(MOTOR_1_PIN_2, 0);
  digitalWrite(MOTOR_2_PIN_1, 1);
  digitalWrite(MOTOR_2_PIN_2, 0);
}

void go_forward() {
  digitalWrite(MOTOR_1_PIN_1, 0);
  digitalWrite(MOTOR_1_PIN_2, 1);
  digitalWrite(MOTOR_2_PIN_1, 1);
  digitalWrite(MOTOR_2_PIN_2, 0);
}

void go_backward() {
  digitalWrite(MOTOR_1_PIN_1, 1);
  digitalWrite(MOTOR_1_PIN_2, 0);
  digitalWrite(MOTOR_2_PIN_1, 0);
  digitalWrite(MOTOR_2_PIN_2, 1);
}

void go_right() {
  digitalWrite(MOTOR_1_PIN_1, 0);
  digitalWrite(MOTOR_1_PIN_2, 1);
  digitalWrite(MOTOR_2_PIN_1, 0);
  digitalWrite(MOTOR_2_PIN_2, 1);
}

void stop() {
  digitalWrite(MOTOR_1_PIN_1, 0);
  digitalWrite(MOTOR_1_PIN_2, 0);
  digitalWrite(MOTOR_2_PIN_1, 0);
  digitalWrite(MOTOR_2_PIN_2, 0);
}

void go(String dir) {
  if (dir.equals("forward")) {
    go_forward();
  } else if (dir.equals("backward")) {
    go_backward();
  } else if (dir.equals("left")) {
    go_left();
  } else if (dir.equals("right")) {
    go_right();
  } else {
    stop();
  }
}

void led_on() {
  digitalWrite(LED_PIN, LED_HIGH);
  // analogWrite(LED_PIN, 255);
}

void led_off() {
  digitalWrite(LED_PIN, LED_LOW);
  // analogWrite(LED_PIN, 0);
}


void action() {
  String direction;
  String uri = server.uri();
  String dest = uri.substring(uri.lastIndexOf("/") + 1);
  Serial.println(dest);
  if (!dest.equals("stop")) {
    if (server.argName(0).equals("delay")) {
      int code = 200;
      // char cont[20];
      if (dest == "left" || dest == "right") {
        go(dest);
        delay(server.arg(0).toInt());
        // float filt = listen_rot(server.arg(0).toInt());
        // sprintf(cont, "%f", filt);
        // cont[19] = '\0';
      } else {
        go(dest);
        delay(server.arg(0).toInt());
      }
      stop();
      // server.send(code, "text/plain", cont);
      server.send(code);
    } else {
      go(dest);
    }
  } else {
    stop();
  }
}

void on_speed() {
  if (server.argName(0).equals("set")) {
    SPEED = server.arg(0).toInt();
    analogWrite(MOTOR_SPEED_PIN_1, SPEED);
    analogWrite(MOTOR_SPEED_PIN_2, SPEED);
  } else {
    char content[15];
    sprintf(content, "{\"speed\": %d}", SPEED);
    content[14] = '\0';
    server.send(200, "text/plain", content);
  }
}

void setup()
{
    Serial.begin(115200);

    pinMode(MOTOR_1_PIN_1, OUTPUT);
    pinMode(MOTOR_1_PIN_2, OUTPUT);
    pinMode(MOTOR_2_PIN_1, OUTPUT);
    pinMode(MOTOR_2_PIN_2, OUTPUT);
    pinMode(MOTOR_SPEED_PIN_1, OUTPUT);
    pinMode(MOTOR_SPEED_PIN_2, OUTPUT);
    pinMode(LED_PIN, OUTPUT);
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, pass);

    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }

    Serial.println("");
    Serial.print("Connected to ");
    Serial.println(ssid);
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    server.on("/go/forward", action);
    server.on("/go/backward", action);
    server.on("/go/left", action);
    server.on("/go/right", action);
    server.on("/go/stop", action);
    server.on("/led/on", led_on);
    server.on("/led/off", led_off);
    server.on("/speed", on_speed);
    server.enableCORS();
    server.begin();
    Serial.println("HTTP server started");

    analogWrite(MOTOR_SPEED_PIN_1, SPEED);
    analogWrite(MOTOR_SPEED_PIN_2, SPEED);
}

void loop()
{
  server.handleClient();
  delay(5);//allow the cpu to switch to other tasks
}
