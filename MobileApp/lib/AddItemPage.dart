import 'dart:io';
import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart'; // ✅ Add this!
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:image_picker/image_picker.dart';

class AddItemPage extends StatefulWidget {
  @override
  _AddItemPageState createState() => _AddItemPageState();
}

class _AddItemPageState extends State<AddItemPage> {
  final _formKey = GlobalKey<FormState>();

  String itemName = '';
  String description = '';
  int quantity = 0;
  double price = 0.0;
  File? imageFile;
  bool _loading = false;

  final picker = ImagePicker();

  Future<void> _pickImage() async {
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      setState(() {
        imageFile = File(picked.path);
      });
    }
  }

  Future<String?> uploadImageToServer(File imageFile) async {
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://10.0.2.2:5000/api/upload'),
      );

      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      var response = await request.send();

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final jsonData = jsonDecode(responseBody);
        final serverUrl = 'http://10.0.2.2:5000';
        final downloadUrl = serverUrl + jsonData['download_url'];
        return downloadUrl;
      } else {
        print('Failed to upload image: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('Error uploading image: $e');
      return null;
    }
  }

  Future<void> _uploadAndSave() async {
    if (!_formKey.currentState!.validate() || imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Please complete all fields and upload an image.')));
      return;
    }

    setState(() {
      _loading = true;
    });

    try {

      User? user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        throw Exception('No user logged in');
      }


      String? imageUrl = await uploadImageToServer(imageFile!);

      if (imageUrl == null) {
        throw Exception('Image upload failed');
      }


      await FirebaseFirestore.instance.collection('items').add({
        'name': itemName,
        'description': description,
        'quantity': quantity,
        'price': price,
        'imageUrl': imageUrl,
        'userId': user.uid,
        'userEmail': user.email,
        'createdAt': Timestamp.now(),
      });

      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Item added successfully.')));

      _formKey.currentState!.reset();
      setState(() {
        imageFile = null;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _loading = false;
      });
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Error: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final accent = Colors.blue.shade800;

    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Text('Add New Item', style: TextStyle(color: accent)),
        centerTitle: true,
        elevation: 1,
        iconTheme: IconThemeData(color: accent),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [

                Text(
                  'Item Details',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: accent,
                  ),
                ),
                SizedBox(height: 16),


                TextFormField(
                  decoration: InputDecoration(
                    labelText: 'Item Name',
                    filled: true,
                    fillColor: Colors.white,
                    border: OutlineInputBorder(),
                  ),
                  validator: (val) =>
                      val == null || val.isEmpty ? 'Enter item name' : null,
                  onChanged: (val) => itemName = val,
                ),
                SizedBox(height: 16),


                TextFormField(
                  decoration: InputDecoration(
                    labelText: 'Description',
                    filled: true,
                    fillColor: Colors.white,
                    border: OutlineInputBorder(),
                  ),
                  maxLines: 3,
                  validator: (val) =>
                      val == null || val.isEmpty ? 'Enter description' : null,
                  onChanged: (val) => description = val,
                ),
                SizedBox(height: 16),


                Row(
                  children: [
                    Expanded(
                      child: TextFormField(
                        decoration: InputDecoration(
                          labelText: 'Quantity',
                          filled: true,
                          fillColor: Colors.white,
                          border: OutlineInputBorder(),
                        ),
                        keyboardType: TextInputType.number,
                        validator: (val) =>
                            val == null || int.tryParse(val) == null
                                ? 'Enter valid quantity'
                                : null,
                        onChanged: (val) => quantity = int.tryParse(val) ?? 0,
                      ),
                    ),
                    SizedBox(width: 12),
                    Expanded(
                      child: TextFormField(
                        decoration: InputDecoration(
                          labelText: 'Price (LKR)',
                          filled: true,
                          fillColor: Colors.white,
                          border: OutlineInputBorder(),
                        ),
                        keyboardType: TextInputType.number,
                        validator: (val) =>
                            val == null || double.tryParse(val) == null
                                ? 'Enter valid price'
                                : null,
                        onChanged: (val) => price = double.tryParse(val) ?? 0.0,
                      ),
                    ),
                  ],
                ),

                SizedBox(height: 24),

                // Section title
                Text(
                  'Upload Image',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: accent,
                  ),
                ),
                SizedBox(height: 16),


                GestureDetector(
                  onTap: _pickImage,
                  child: Container(
                    height: 200,
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey.shade400),
                    ),
                    child: imageFile != null
                        ? ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.file(
                              imageFile!,
                              fit: BoxFit.cover,
                            ),
                          )
                        : Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(Icons.add_a_photo,
                                    size: 50, color: Colors.grey),
                                SizedBox(height: 8),
                                Text('Tap to upload image',
                                    style: TextStyle(color: Colors.grey)),
                              ],
                            ),
                          ),
                  ),
                ),

                SizedBox(height: 30),

                // Submit Button
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: _loading ? null : _uploadAndSave,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: accent,
                      padding: EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    child: _loading
                        ? CircularProgressIndicator(color: Colors.white)
                        : Text(
                            'Save Item',
                            style: TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.bold),
                          ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
