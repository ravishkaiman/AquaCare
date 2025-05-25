import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_animate/flutter_animate.dart';

class DiseasePredictPage extends StatefulWidget {
  @override
  _DiseasePredictPageState createState() => _DiseasePredictPageState();
}

class _DiseasePredictPageState extends State<DiseasePredictPage>
    with SingleTickerProviderStateMixin {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _temperatureController = TextEditingController();
  final TextEditingController _humidityController = TextEditingController();
  final TextEditingController _phController = TextEditingController();

  String _ventilation = 'low';
  String _lightIntensity = 'low';

  String? _prediction;
  bool _loading = false;

  Future<void> _predictDisease() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _loading = true;
      _prediction = null;
    });

    final payload = {
      "temperature": double.parse(_temperatureController.text),
      "humidity": double.parse(_humidityController.text),
      "ventilation": _ventilation,
      "light_intensity": _lightIntensity,
      "ph": double.parse(_phController.text),
    };

    try {
      final response = await http.post(
        Uri.parse("http://10.0.2.2:5000/disease_api/predict"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(payload),
      );

      final result = jsonDecode(response.body);
      setState(() {
        _prediction = result["prediction"] ?? "Error in prediction";
      });
    } catch (e) {
      setState(() {
        _prediction = "Error: $e";
      });
    } finally {
      setState(() {
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("🌿 Disease Predictor"),
        centerTitle: true,
        elevation: 6,
        automaticallyImplyLeading: false, // 🔒 Hides the back button
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Animate(
          effects: [FadeEffect(duration: 600.ms)],
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("Enter Plant Conditions",
                    style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.green[800])),
                SizedBox(height: 20),
                _buildTextField(_temperatureController, "🌡️ Temperature"),
                _buildTextField(_humidityController, "💧 Humidity"),
                _buildTextField(_phController, "🧪 pH Level"),
                DropdownButtonFormField<String>(
                  value: _ventilation,
                  decoration: InputDecoration(labelText: "💨 Ventilation"),
                  items: ["low", "high"]
                      .map((v) => DropdownMenuItem(value: v, child: Text(v)))
                      .toList(),
                  onChanged: (value) => setState(() => _ventilation = value!),
                ),
                SizedBox(height: 16),
                DropdownButtonFormField<String>(
                  value: _lightIntensity,
                  decoration: InputDecoration(labelText: "💡 Light Intensity"),
                  items: ["low", "medium", "high"]
                      .map((v) => DropdownMenuItem(value: v, child: Text(v)))
                      .toList(),
                  onChanged: (value) =>
                      setState(() => _lightIntensity = value!),
                ),
                SizedBox(height: 30),
                Center(
                  child: AnimatedContainer(
                    duration: Duration(milliseconds: 300),
                    width: _loading ? 70 : 200,
                    height: 50,
                    child: ElevatedButton(
                      onPressed: _loading ? null : _predictDisease,
                      child: _loading
                          ? SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2.5,
                                color: Colors.white,
                              ))
                          : Text("Predict Disease"),
                    ).animate().fadeIn().slideY(),
                  ),
                ),
                SizedBox(height: 30),
                if (_prediction != null)
                  Container(
                    padding: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.green[50],
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.green),
                    ),
                    child: Row(
                      children: [
                        Icon(Icons.health_and_safety,
                            color: Colors.green, size: 28),
                        SizedBox(width: 10),
                        Expanded(
                          child: Text("Prediction: $_prediction",
                              style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.green[900])),
                        )
                      ],
                    ),
                  ).animate().fadeIn().scale(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(TextEditingController controller, String label) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: TextFormField(
        controller: controller,
        decoration: InputDecoration(labelText: label),
        keyboardType: TextInputType.number,
        validator: (value) =>
            value == null || value.isEmpty ? "Required" : null,
      ),
    );
  }
}
