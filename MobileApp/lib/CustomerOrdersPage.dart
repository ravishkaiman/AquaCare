

import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:intl/intl.dart';

class CustomerOrdersPage extends StatelessWidget {
  const CustomerOrdersPage({super.key});

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

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) {
      return const Scaffold(
        body: Center(child: Text('Please log in to view your orders.')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('My Purchases'),
        backgroundColor: Colors.blue.shade700,
      ),
      body: StreamBuilder<QuerySnapshot>(

        stream: FirebaseFirestore.instance
            .collection('orders')
            .where('buyerEmail', isEqualTo: user.email)
            .orderBy('timestamp', descending: true)
            .snapshots(),
        builder: (ctx, snap) {
          if (snap.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          final docs = snap.data?.docs ?? [];
          if (docs.isEmpty) {
            return const Center(child: Text('You have no orders yet.'));
          }

          return ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: docs.length,
            itemBuilder: (_, i) {
              final data = docs[i].data()! as Map<String, dynamic>;
              final ts = (data['timestamp'] as Timestamp).toDate();
              final formattedTs = DateFormat.yMMMd().add_jm().format(ts);
              final prefDate = (data['preferredDate'] as Timestamp?)?.toDate();
              final formattedPref =
                  prefDate != null ? DateFormat.yMMMd().format(prefDate) : '—';

              final status = data['status'] as String? ?? 'Pending';
              final statusColor = _statusColor(status);

              return Card(
                margin: const EdgeInsets.only(bottom: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                elevation: 3,
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [

                      Row(
                        children: [
                          Expanded(
                            child: Text(
                              data['productName'] ?? 'Product',
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
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


                      Text('Price: Rs. ${data['price']}',
                          style: const TextStyle(fontSize: 14)),
                      Text('Ordered on: $formattedTs',
                          style: const TextStyle(
                              fontSize: 12, color: Colors.grey)),
                      const Divider(height: 24),


                      Text('Delivery Address:',
                          style: const TextStyle(fontWeight: FontWeight.w600)),
                      Text(data['deliveryAddress'] ?? '—'),
                      const SizedBox(height: 8),
                      Text('Phone:',
                          style: const TextStyle(fontWeight: FontWeight.w600)),
                      Text(data['phone'] ?? '—'),
                      const SizedBox(height: 8),
                      Text('Preferred Date:',
                          style: const TextStyle(fontWeight: FontWeight.w600)),
                      Text(formattedPref),
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
