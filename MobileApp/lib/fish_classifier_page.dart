import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';

class FishClassifierPage extends StatefulWidget {
  @override
  _FishClassifierPageState createState() => _FishClassifierPageState();
}

class _FishClassifierPageState extends State<FishClassifierPage>
    with SingleTickerProviderStateMixin {
  File? _image;
  String? _result;
  bool _loading = false;

  final _picker = ImagePicker();

  late AnimationController _animController;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: Duration(milliseconds: 600),
    );
    _fadeAnim = CurvedAnimation(
      parent: _animController,
      curve: Curves.easeIn,
    );
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  Future<void> _pick(ImageSource src) async {
    final perm = src == ImageSource.camera
        ? await Permission.camera.request()
        : await Permission.storage.request();
    if (!perm.isGranted) {
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Permission denied')));
      return;
    }

    final picked = await _picker.pickImage(source: src);
    if (picked != null) {
      setState(() {
        _image = File(picked.path);
        _result = null;
      });
      _animController.reset();
    }
  }

  Future<void> _classify() async {
    if (_image == null) return;

    setState(() {
      _loading = true;
      _result = null;
    });

    final uri = Uri.parse("http://10.0.2.2:5000/predict");
    final req = http.MultipartRequest('POST', uri)
      ..files.add(
        await http.MultipartFile.fromPath('image', _image!.path),
      );

    try {
      final res = await req.send();
      final data = jsonDecode(await res.stream.bytesToString());
      setState(() {
        _result = res.statusCode == 200
            ? data['predicted_class']
            : data['error'] ?? 'Unknown error';
      });
      _animController.forward();
    } catch (e) {
      setState(() => _result = "Error: $e");
      _animController.forward();
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 1,
        centerTitle: true,
        title: Text(
          'Fish Classifier',
          style: TextStyle(
            color: Colors.blueGrey[900],
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            AnimatedContainer(
              duration: Duration(milliseconds: 500),
              curve: Curves.easeInOut,
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    blurRadius: 8,
                    offset: Offset(0, 4),
                  ),
                ],
              ),
              child: Column(
                children: [
                  _image != null
                      ? ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: Image.file(_image!, height: 200),
                        )
                      : Column(
                          children: [
                            Icon(Icons.add_photo_alternate,
                                size: 80, color: Colors.grey[400]),
                            SizedBox(height: 8),
                            Text(
                              'Upload a fish image to classify',
                              style: TextStyle(color: Colors.grey[600]),
                            )
                          ],
                        ),
                  SizedBox(height: 20),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _loading
                              ? null
                              : () => _pick(ImageSource.gallery),
                          icon: Icon(Icons.photo, color: Colors.white),
                          label: Text('Gallery'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blue,
                            padding: EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                        ),
                      ),
                      SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed:
                              _loading ? null : () => _pick(ImageSource.camera),
                          icon: Icon(Icons.camera_alt, color: Colors.white),
                          label: Text('Camera'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blue,
                            padding: EdgeInsets.symmetric(vertical: 14),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            SizedBox(height: 20),
            if (_image != null)
              ElevatedButton(
                onPressed: _loading ? null : _classify,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  padding: EdgeInsets.symmetric(vertical: 16, horizontal: 32),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: _loading
                    ? SizedBox(
                        height: 24,
                        width: 24,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor:
                              AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : Text(
                        'Classify Fish',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
              ),
            SizedBox(height: 20),
            SizeTransition(
              sizeFactor: _fadeAnim,
              axisAlignment: -1.0,
              child: _result != null
                  ? Container(
                      width: double.infinity,
                      padding: EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.grey.withOpacity(0.2),
                            blurRadius: 8,
                            offset: Offset(0, 4),
                          ),
                        ],
                      ),
                      child: Column(
                        children: [
                          Icon(Icons.check_circle_outline,
                              color: Colors.blue, size: 40),
                          SizedBox(height: 12),
                          Text(
                            'Prediction Result:',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w500,
                              color: Colors.grey[700],
                            ),
                          ),
                          SizedBox(height: 8),
                          Text(
                            _result!,
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              color: Colors.deepPurple,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ],
                      ),
                    )
                  : SizedBox.shrink(),
            ),
          ],
        ),
      ),
    );
  }
}
