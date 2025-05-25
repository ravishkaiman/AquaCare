import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:my_app/LoginPage.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key});

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  String userEmail = 'Loading...';
  String userName = 'Loading...';
  int reminderCount = 0;

  @override
  void initState() {
    super.initState();
    _loadSettingsData();
  }

  Future<void> _loadSettingsData() async {
    final currentUser = FirebaseAuth.instance.currentUser;
    final prefs = await SharedPreferences.getInstance();
    final reminders = prefs.getStringList('reminders') ?? [];

    if (currentUser != null) {
      final doc = await FirebaseFirestore.instance
          .collection('users')
          .doc(currentUser.uid)
          .get();

      setState(() {
        reminderCount = reminders.length;
        userEmail = currentUser.email ?? "No email";
        userName = doc.data()?['name'] ?? "Unnamed User";
      });
    }
  }

  Future<void> _logout() async {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const Center(child: CircularProgressIndicator()),
    );

    try {
      await FirebaseAuth.instance.signOut();
      final prefs = await SharedPreferences.getInstance();
      await prefs.clear();

      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const LoginPage()),
      );
    } catch (e) {
      Navigator.of(context).pop();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Logout failed")),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final accent = Color(0xFF1565C0); // deep blue accent

    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        automaticallyImplyLeading: false, // ← this hides the back button

        backgroundColor: Colors.white,
        elevation: 1,
        centerTitle: true,
        title: Text(
          'Settings',
          style: TextStyle(color: accent, fontWeight: FontWeight.bold),
        ),
        iconTheme: IconThemeData(color: accent),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 24),
        children: [
          // --- Account Section ---
          Text('Account',
              style: TextStyle(
                color: Colors.grey[700],
                fontSize: 14,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.2,
              )),
          const SizedBox(height: 8),
          Card(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            elevation: 4,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  CircleAvatar(
                    radius: 32,
                    backgroundColor: accent,
                    child: Icon(Icons.person, size: 32, color: Colors.white),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(userName,
                            style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: accent)),
                        const SizedBox(height: 4),
                        Text(userEmail,
                            style: TextStyle(color: Colors.grey[600])),
                      ],
                    ),
                  ),
                  Icon(Icons.edit, color: accent),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          // --- Preferences Section ---
          Text('Preferences',
              style: TextStyle(
                color: Colors.grey[700],
                fontSize: 14,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.2,
              )),
          const SizedBox(height: 8),
          Card(
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            elevation: 2,
            child: ListTile(
              leading: Icon(Icons.alarm, color: accent, size: 28),
              title: Text("Active Reminders",
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: Colors.grey[800])),
              trailing: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: accent.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text('$reminderCount',
                    style: TextStyle(
                        color: accent,
                        fontWeight: FontWeight.bold,
                        fontSize: 16)),
              ),
            ),
          ),
          const SizedBox(height: 24),

          // (You can add more preference items here as additional ListTiles)

          // --- Actions Section ---
          Text('Actions',
              style: TextStyle(
                color: Colors.grey[700],
                fontSize: 14,
                fontWeight: FontWeight.w600,
                letterSpacing: 1.2,
              )),
          const SizedBox(height: 8),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              icon: Icon(
                Icons.logout,
                color: Colors.white, // <-- white color for the icon
              ),
              label: Text(
                "Logout",
                style: TextStyle(
                  color: Colors.white, // <-- white text color
                  fontWeight: FontWeight.bold, // optional: make it bold
                  fontSize: 16, // optional: control size
                ),
              ),
              onPressed: _logout,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red[600],
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30)),
                textStyle:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
