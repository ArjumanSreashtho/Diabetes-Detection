#include "eml_net.h"
#include "rfc_diabetes.h"
//For Artificial Neural Network
static const float diabetes_model_layer0_weights[96] = { -0.883749f, -1.828300f, 0.254916f, 1.203327f, 2.921104f, 0.595970f, 0.771837f, -0.835006f, -0.818809f, -0.283822f, 0.331837f, 0.005788f, -0.076783f, -0.101782f, 0.510586f, 0.407546f, -0.814550f, -0.414880f, -0.066051f, -0.056761f, -0.263104f, -0.317449f, -0.345990f, 0.177002f, -0.374953f, 0.166836f, -0.250094f, -0.627402f, 0.073300f, 0.157851f, 0.515486f, 0.279633f, 0.484492f, -0.292949f, -0.161728f, -0.020119f, 0.442325f, 0.198423f, -0.269516f, -0.074259f, -0.272912f, -0.610546f, -0.009217f, 0.051834f, 0.005071f, 0.091673f, -0.378228f, 0.284459f, -0.048385f, 0.244737f, -0.376868f, -0.780014f, -0.689628f, 0.071340f, -0.291137f, 0.461962f, -0.137799f, -0.110734f, 0.200234f, 0.483624f, 0.083262f, -0.735249f, -0.282207f, 0.007620f, 0.252736f, 0.133410f, -0.300379f, 0.098330f, 0.619199f, -0.085341f, -0.701355f, 0.626310f, 0.112672f, -0.263981f, -2.673459f, -0.416844f, 0.123326f, -0.302257f, -1.188759f, -1.195794f, -0.729464f, 0.221728f, 1.458155f, 0.950003f, 0.790462f, 0.538141f, -0.450201f, 0.079165f, 0.264714f, 0.557592f, 0.119344f, -0.273106f, 0.649227f, 0.131486f, -0.550325f, -0.123239f };
static const float diabetes_model_layer0_biases[12] = { 2.029392f, -1.938364f, -1.188876f, 0.231395f, -0.811941f, -0.174513f, 2.541039f, 3.620552f, -0.344983f, 0.000000f, -0.338021f, -3.311292f };
static const float diabetes_model_layer1_weights[96] = { 0.040969f, 0.369321f, -0.603477f, 0.505954f, -0.177908f, 0.022142f, -0.589655f, -0.581151f, -0.721850f, -0.128086f, 0.362877f, -1.208362f, -0.035690f, 0.410207f, 0.432454f, 0.257455f, 0.378764f, -1.349013f, 0.037998f, 0.116903f, -0.299597f, -0.323857f, -0.239594f, -0.391749f, -0.416542f, -0.423466f, -0.300950f, 0.101376f, 0.356615f, 0.034838f, -0.569692f, 0.530450f, 1.452886f, -0.487128f, 0.254779f, -0.617364f, -0.403130f, 0.378674f, -0.506676f, 1.225911f, -0.787147f, -0.447902f, 0.366311f, 0.835962f, -0.443821f, 0.282433f, 0.765722f, 1.263634f, -0.169940f, -0.242099f, 0.183112f, 0.294078f, -0.316198f, 0.177093f, 0.518771f, -0.672435f, -0.939099f, -0.210166f, 0.209892f, 0.598541f, 0.178147f, -0.462538f, -0.222251f, -0.484966f, 0.252298f, -0.921365f, -0.260726f, -0.071772f, -0.208347f, -0.565096f, -0.287838f, 0.135245f, -0.205404f, 0.433342f, 0.415437f, 0.296975f, -0.030629f, 0.428326f, 0.145141f, 0.248985f, -0.165460f, -0.612031f, 0.007210f, 0.187274f, 0.046503f, 0.171077f, -0.136645f, -0.540082f, 0.286587f, 0.200708f, -0.626687f, -0.508306f, -0.349175f, 0.092052f, -0.852209f, 0.331431f };
static const float diabetes_model_layer1_biases[8] = { -2.505902f, -1.009623f, -0.271015f, 1.437911f, -0.182938f, -0.098380f, -0.323320f, -3.346967f };
static const float diabetes_model_layer2_weights[8] = { 0.379623f, 1.714468f, 0.524556f, -1.755432f, 0.469948f, 0.411078f, 1.887663f, 0.552144f };
static const float diabetes_model_layer2_biases[1] = { -3.591292f };
static float diabetes_model_buf1[12];
static float diabetes_model_buf2[12];
static const EmlNetLayer diabetes_model_layers[3] = {
  { 12, 8, diabetes_model_layer0_weights, diabetes_model_layer0_biases, EmlNetActivationRelu },
  { 8, 12, diabetes_model_layer1_weights, diabetes_model_layer1_biases, EmlNetActivationRelu },
  { 1, 8, diabetes_model_layer2_weights, diabetes_model_layer2_biases, EmlNetActivationLogistic }
};
static EmlNet diabetes_model = { 3, diabetes_model_layers, diabetes_model_buf1, diabetes_model_buf2, 12 };

void setup() {
  Serial.begin(115200);

}

void loop() {
  float features[8];
  String classNames[] = {"False", "True"};
  int cnt = 0;
  bool ck = 0;
  String number = "";
  if (Serial.available() > 0)
  {
    while (true)
    {
      char input = Serial.read();
      if (input == '\n')
      {
        ck = 1;
        features[cnt] = number.toFloat();
        Serial.println(number);
        break;
      }
      if (input == ' ' || input == ',')
      {
        features[cnt] = number.toFloat();
        Serial.println(number);
        number = "";
        cnt++;
        continue;
      }
      number += input;
    }
  }
  cnt = 1;
  if (ck == 1)
  {
    int result  = eml_net_predict(&diabetes_model, features, 8);
    Serial.print("Using Artificial Neural Network: ");
    Serial.println(result);

    result  = rfc_diabetes_predict(features, 8);
    Serial.print("Using Random Forest Tree: ");
    Serial.println(result);
    cnt++;

  }
  delay(10);
}
