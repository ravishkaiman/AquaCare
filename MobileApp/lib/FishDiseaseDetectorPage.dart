import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

class FishDiseaseDetectorPage extends StatefulWidget {
  @override
  _FishDiseaseDetectorPageState createState() =>
      _FishDiseaseDetectorPageState();
}

class _FishDiseaseDetectorPageState extends State<FishDiseaseDetectorPage> {
  File? _image;
  String? _prediction;
  bool _loading = false;

  Future<void> _pickImage(ImageSource src) async {
    final perm = src == ImageSource.camera
        ? await Permission.camera.request()
        : await Permission.storage.request();
    if (!perm.isGranted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Permission denied')),
      );
      return;
    }
    final picked = await ImagePicker().pickImage(source: src);
    if (picked != null) {
      setState(() {
        _image = File(picked.path);
        _prediction = null;
      });
    }
  }

  Future<void> _predict() async {
    if (_image == null) return;
    setState(() {
      _loading = true;
      _prediction = null;
    });

    final uri = Uri.parse("http://10.0.2.2:5000/api/predict");
    final req = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('file', _image!.path));

    try {
      final res = await req.send();
      final data = jsonDecode(await res.stream.bytesToString());
      setState(() {
        _prediction = res.statusCode == 200
            ? data['prediction']
            : data['error'] ?? 'Error';
      });
    } catch (e) {
      setState(() => _prediction = "Error: $e");
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final accent = Colors.blue.shade800;

    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        automaticallyImplyLeading: false,
        backgroundColor: Colors.white,
        elevation: 1,
        centerTitle: true,
        iconTheme: IconThemeData(color: accent),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.local_hospital, color: accent),
            SizedBox(width: 8),
            Text(
              'Fish Disease Detector',
              style: TextStyle(color: accent, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [

              Card(
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
                elevation: 4,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text(
                        'Upload Fish Image',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: accent,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(
                        'Pick an image from gallery or camera to start detection.',
                        style: TextStyle(color: Colors.grey[600]),
                      ),
                      SizedBox(height: 16),
                      GestureDetector(
                        onTap: _loading
                            ? null
                            : () => _pickImage(ImageSource.gallery),
                        child: Container(
                          height: 200,
                          decoration: BoxDecoration(
                            color: Colors.grey[100],
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Colors.grey[300]!),
                            image: _image != null
                                ? DecorationImage(
                                    image: FileImage(_image!),
                                    fit: BoxFit.cover)
                                : null,
                          ),
                          child: _image == null
                              ? Center(
                                  child: Icon(
                                    Icons.add_photo_alternate,
                                    size: 60,
                                    color: Colors.grey[400],
                                  ),
                                )
                              : null,
                        ),
                      ),
                      SizedBox(height: 20),
                      Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: _loading
                                  ? null
                                  : () => _pickImage(ImageSource.gallery),
                              icon: Icon(Icons.photo, color: Colors.white),
                              label: Text('Gallery',
                                  style: TextStyle(color: Colors.white)),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: accent,
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(8)),
                                padding: EdgeInsets.symmetric(vertical: 14),
                              ),
                            ),
                          ),
                          SizedBox(width: 12),
                          Expanded(
                            child: ElevatedButton.icon(
                              onPressed: _loading
                                  ? null
                                  : () => _pickImage(ImageSource.camera),
                              icon: Icon(Icons.camera_alt, color: Colors.white),
                              label: Text('Camera',
                                  style: TextStyle(color: Colors.white)),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: accent,
                                shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(8)),
                                padding: EdgeInsets.symmetric(vertical: 14),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),

              SizedBox(height: 20),

              // Section 2 - Predict Button
              if (_image != null)
                Card(
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  elevation: 4,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: ElevatedButton.icon(
                      onPressed: _loading ? null : _predict,
                      icon: Icon(Icons.search, color: Colors.white),
                      label: Text(
                        _loading ? 'Detecting...' : 'Detect Disease',
                        style: TextStyle(color: Colors.white, fontSize: 16),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: accent,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8)),
                        padding: EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                ),

              SizedBox(height: 20),

              // Section 3 - Prediction Result
              if (_prediction != null)
                Card(
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  elevation: 4,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Icon(Icons.verified, color: accent, size: 30),
                        SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _prediction!,
                            style: TextStyle(
                                fontSize: 18, fontWeight: FontWeight.w600),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
