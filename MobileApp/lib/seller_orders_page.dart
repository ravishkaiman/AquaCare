// seller_orders_page.dart

import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:intl/intl.dart';

class SellerOrdersPage extends StatefulWidget {
  const SellerOrdersPage({super.key});

  @override
  State<SellerOrdersPage> createState() => _SellerOrdersPageState();
}

class _SellerOrdersPageState extends State<SellerOrdersPage> {
  final _firestore = FirebaseFirestore.instance;
  final _auth = FirebaseAuth.instance;

  List<String>? _myProductIds;

  @override
  void initState() {
    super.initState();
    _loadMyProductIds();
  }

  Future<void> _loadMyProductIds() async {
    final user = _auth.currentUser;
    if (user == null) return;
    final snap = await _firestore
        .collection('items')
        .where('userId', isEqualTo: user.uid)
        .get();
    setState(() {
      _myProductIds = snap.docs.map((d) => d.id).toList();
    });
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'Delivered':
        return Colors.green;
      case 'Canceled':
        return Colors.red;
      default:
        return Colors.orange;
    }
  }

  void _updateStatus(String orderId, String newStatus) {
    _firestore.collection('orders').doc(orderId).update({
      'status': newStatus,
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Order marked $newStatus')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = _auth.currentUser;
    if (user == null) {
      return const Scaffold(
        body: Center(child: Text('Please log in as a seller.')),
      );
    }

    if (_myProductIds == null) {
      return Scaffold(
        appBar: AppBar(title: Text('Seller Orders')),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_myProductIds!.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('Seller Orders')),
        body: const Center(child: Text('You have no products listed.')),
      );
    }

    // Firestore supports whereIn only up to 10 items. For >10 you'll need another approach.
    final query = _firestore
        .collection('orders')
        .where('productId', whereIn: _myProductIds!)
        .orderBy('timestamp', descending: true);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Manage Customer Orders'),
        backgroundColor: Colors.blue.shade700,
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: query.snapshots(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          final orders = snap.data!.docs;
          if (orders.isEmpty) {
            return const Center(child: Text('No orders yet.'));
          }
          return ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: orders.length,
            itemBuilder: (context, i) {
              final doc = orders[i];
              final data = doc.data()! as Map<String, dynamic>;

              final ts = (data['timestamp'] as Timestamp).toDate();
              final orderedOn =
                  DateFormat.yMMMd().add_jm().format(ts.toLocal());

              final prefTs = data['preferredDate'] as Timestamp?;
              final prefOn = prefTs != null
                  ? DateFormat.yMMMd().format(prefTs.toDate().toLocal())
                  : '–';

              final status = (data['status'] as String?) ?? 'Pending';
              final statusColor = _statusColor(status);

              return Card(
                margin: const EdgeInsets.only(bottom: 16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12)),
                elevation: 3,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Header row: product & status
                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              data['productName'] ?? 'Product',
                              style: const TextStyle(
                                  fontSize: 18, fontWeight: FontWeight.bold),
                            ),
                          ),
                          Chip(
                            label: Text(status),
                            backgroundColor: statusColor.withOpacity(0.2),
                            labelStyle: TextStyle(color: statusColor),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),

                      // Buyer info & pricing
                      Text('Buyer: ${data['buyerName']}'),
                      Text('Email: ${data['buyerEmail']}'),
                      Text('Price: Rs. ${data['price']}'),
                      const SizedBox(height: 8),

                      // Delivery details
                      Text('Address: ${data['deliveryAddress']}'),
                      Text('Phone: ${data['phone']}'),
                      Text('Deliver by: $prefOn'),
                      const SizedBox(height: 8),

                      // Timestamps
                      Text('Ordered on: $orderedOn',
                          style: const TextStyle(
                              fontSize: 12, color: Colors.grey)),
                      const SizedBox(height: 12),

                      // Actions
                      if (status == 'Pending')
                        Row(
                          children: [
                            TextButton(
                              onPressed: () =>
                                  _updateStatus(doc.id, 'Delivered'),
                              child: const Text('Mark Delivered'),
                            ),
                            const SizedBox(width: 16),
                            TextButton(
                              onPressed: () =>
                                  _updateStatus(doc.id, 'Canceled'),
                              child: const Text(
                                'Cancel',
                                style: TextStyle(color: Colors.red),
                              ),
                            ),
                          ],
                        ),
                    ],
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}
