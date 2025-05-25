import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

// import your pages here:
import 'package:my_app/AddItemPage.dart';
import 'package:my_app/ManageProductsPage.dart';
import 'package:my_app/CustomerOrdersPage.dart';
import 'package:my_app/marketplace_page.dart';
import 'package:my_app/seller_orders_page.dart';

class ManageListingsPage extends StatefulWidget {
  @override
  _ManageListingsPageState createState() => _ManageListingsPageState();
}

class _ManageListingsPageState extends State<ManageListingsPage> {
  final accent = Colors.blue.shade800;
  String? adminName;

  @override
  void initState() {
    super.initState();
    fetchAdminName();
  }

  Future<void> fetchAdminName() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      final doc = await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .get();
      setState(() {
        adminName = doc.data()?['name'] ?? 'Admin';
      });
    } else {
      setState(() {
        adminName = 'Admin';
      });
    }
  }

  Widget _buildButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return ElevatedButton.icon(
      onPressed: onTap,
      icon: Icon(icon, color: Colors.white),
      label: Text(
        label,
        style: const TextStyle(
            fontSize: 16, fontWeight: FontWeight.w600, color: Colors.white),
      ),
      style: ElevatedButton.styleFrom(
        backgroundColor: accent,
        padding: const EdgeInsets.symmetric(vertical: 16),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: Colors.white,
        title: Text('Manage Listings', style: TextStyle(color: accent)),
        centerTitle: true,
        elevation: 1,
        iconTheme: IconThemeData(color: accent),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 30),
              Text(
                adminName == null ? 'Welcome 👋' : 'Welcome $adminName 👋',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: accent,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              Text(
                'Manage your listings and orders easily.',
                style: TextStyle(fontSize: 16, color: Colors.grey[700]),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 40),

              Text(
                'Actions',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.grey[800],
                ),
              ),
              const SizedBox(height: 16),

              // 1. Add New Item
              _buildButton(
                icon: Icons.add_circle_outline,
                label: 'Add New Item',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => AddItemPage()),
                ),
              ),
              const SizedBox(height: 16),


              _buildButton(
                icon: Icons.inventory,
                label: 'Update Listings',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => ManageProductsPage()),
                ),
              ),
              const SizedBox(height: 16),


              _buildButton(
                icon: Icons.shopping_bag,
                label: 'Seller Orders',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const SellerOrdersPage()),
                ),
              ),
              const SizedBox(height: 16),


              _buildButton(
                icon: Icons.receipt_long,
                label: 'My Purchases',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CustomerOrdersPage()),
                ),
              ),
              const SizedBox(height: 16),


              _buildButton(
                icon: Icons.storefront,
                label: 'Marketplace',
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const MarketplacePage()),
                ),
              ),
              const SizedBox(height: 40),


              Center(
                child: Text(
                  'AquaCare Management © 2025',
                  style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                ),
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}
