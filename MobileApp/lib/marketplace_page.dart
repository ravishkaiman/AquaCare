// marketplace_page.dart

import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:intl/intl.dart';

class MarketplacePage extends StatefulWidget {
  const MarketplacePage({super.key});

  @override
  State<MarketplacePage> createState() => _MarketplacePageState();
}

class _MarketplacePageState extends State<MarketplacePage> {
  final _auth = FirebaseAuth.instance;
  final _firestore = FirebaseFirestore.instance;

  String _searchText = '';
  final _searchController = TextEditingController();

  final currencyFormat = NumberFormat.currency(locale: 'en_LK', symbol: 'Rs. ');
  final Color primary = Colors.blue.shade700;

  bool _isMine(Map<String, dynamic> d, String? uid) => d['userId'] == uid;

  @override
  Widget build(BuildContext context) {
    final userId = _auth.currentUser?.uid;

    return DefaultTabController(
      length: 4,
      child: Scaffold(
        backgroundColor: Colors.grey[100],
        appBar: AppBar(
          backgroundColor: primary,
          title: const Text('Marketplace'),
          bottom: TabBar(
            labelColor: Colors.white, // active tab text
            unselectedLabelColor: Colors.white70, // inactive tab text
            indicatorColor: Colors.white, // the little underline
            tabs: const [
              Tab(text: 'All'),
              Tab(text: 'Low Price'),
              Tab(text: 'Today'),
              Tab(text: 'Latest'),
            ],
          ),
        ),
        body: Column(
          children: [
            // Search
            Padding(
              padding: const EdgeInsets.all(12),
              child: TextField(
                controller: _searchController,
                decoration: InputDecoration(
                  hintText: 'Search products...',
                  prefixIcon: const Icon(Icons.search),
                  filled: true,
                  fillColor: Colors.white,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide.none,
                  ),
                ),
                onChanged: (v) => setState(() => _searchText = v.toLowerCase()),
              ),
            ),

            // Tab views
            Expanded(
              child: TabBarView(
                children: [
                  _buildGrid(
                    stream: _firestore
                        .collection('items')
                        .orderBy('name')
                        .snapshots(),
                    userId: userId,
                    todayOnly: false,
                  ),
                  _buildGrid(
                    stream: _firestore
                        .collection('items')
                        .orderBy('price')
                        .snapshots(),
                    userId: userId,
                    todayOnly: false,
                  ),
                  _buildGrid(
                    stream: _firestore
                        .collection('items')
                        .orderBy('createdAt', descending: true)
                        .snapshots(),
                    userId: userId,
                    todayOnly: true,
                  ),
                  _buildGrid(
                    stream: _firestore
                        .collection('items')
                        .orderBy('createdAt', descending: true)
                        .snapshots(),
                    userId: userId,
                    todayOnly: false,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildGrid({
    required Stream<QuerySnapshot> stream,
    required String? userId,
    required bool todayOnly,
  }) {
    return StreamBuilder<QuerySnapshot>(
      stream: stream,
      builder: (ctx, snap) {
        if (!snap.hasData)
          return const Center(child: CircularProgressIndicator());

        final docs = snap.data!.docs.where((doc) {
          final d = doc.data()! as Map<String, dynamic>;
          if (_isMine(d, userId)) return false;
          final name = (d['name'] as String?)?.toLowerCase() ?? '';
          if (!name.contains(_searchText)) return false;
          if (todayOnly) {
            final ts = (d['createdAt'] as Timestamp?)?.toDate();
            final now = DateTime.now();
            if (ts == null ||
                ts.year != now.year ||
                ts.month != now.month ||
                ts.day != now.day) return false;
          }
          return true;
        }).toList();

        if (docs.isEmpty)
          return const Center(child: Text('No products found.'));

        return GridView.builder(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            mainAxisSpacing: 16,
            crossAxisSpacing: 16,
            childAspectRatio: 0.60, // taller cards
          ),
          itemCount: docs.length,
          itemBuilder: (ctx, i) {
            final doc = docs[i];
            final d = doc.data()! as Map<String, dynamic>;
            return _buildCard(d, doc.id);
          },
        );
      },
    );
  }

  Widget _buildCard(Map<String, dynamic> d, String docId) {
    final postedDate = (d['createdAt'] as Timestamp).toDate();
    final postedStr = DateFormat.yMMMd().format(postedDate);
    return GestureDetector(
      onTap: () => _showOrderSheet(d, docId),
      child: Card(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        elevation: 4,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Image
            ClipRRect(
              borderRadius:
                  const BorderRadius.vertical(top: Radius.circular(16)),
              child: Image.network(
                d['imageUrl'] ?? '',
                height: 140,
                width: double.infinity,
                fit: BoxFit.cover,
                errorBuilder: (c, e, st) => Container(
                  height: 140,
                  color: Colors.grey[300],
                  child: const Icon(Icons.broken_image,
                      size: 40, color: Colors.grey),
                ),
              ),
            ),

            // Info
            Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(d['name'] ?? '',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: primary)),
                  const SizedBox(height: 6),
                  Text(currencyFormat.format(d['price']),
                      style: const TextStyle(
                          fontSize: 14, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 4),
                  Text('Qty: ${d['quantity']}',
                      style: const TextStyle(fontSize: 12)),
                  const SizedBox(height: 4),
                  Text('Posted: $postedStr',
                      style: const TextStyle(fontSize: 10, color: Colors.grey)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showOrderSheet(Map<String, dynamic> data, String productId) {
    final addressCtrl = TextEditingController();
    final phoneCtrl = TextEditingController();
    DateTime? selectedDate;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => Padding(
        padding: EdgeInsets.only(
          left: 20,
          right: 20,
          bottom: MediaQuery.of(context).viewInsets.bottom + 20,
          top: 24,
        ),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Confirm Purchase',
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 16),
              TextField(
                controller: addressCtrl,
                decoration: const InputDecoration(
                  labelText: 'Delivery Address',
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: phoneCtrl,
                decoration: const InputDecoration(
                  labelText: 'Phone Number',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.phone,
              ),
              const SizedBox(height: 12),
              StatefulBuilder(builder: (ctx, setState) {
                return TextButton.icon(
                  onPressed: () async {
                    final tomorrow =
                        DateTime.now().add(const Duration(days: 1));
                    final picked = await showDatePicker(
                      context: ctx,
                      initialDate: tomorrow,
                      firstDate: tomorrow,
                      lastDate: DateTime(2100),
                    );
                    if (picked != null) setState(() => selectedDate = picked);
                  },
                  icon: const Icon(Icons.calendar_today),
                  label: Text(
                    selectedDate == null
                        ? 'Select Delivery Date'
                        : 'Delivery: ${DateFormat.yMMMd().format(selectedDate!)}',
                  ),
                );
              }),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () async {
                  if (addressCtrl.text.isEmpty ||
                      phoneCtrl.text.isEmpty ||
                      selectedDate == null) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                          content: Text('Fill all fields and choose a date.')),
                    );
                    return;
                  }
                  final user = _auth.currentUser;
                  if (user == null) return;

                  await _firestore.collection('orders').add({
                    'productId': productId,
                    'productName': data['name'],
                    'price': data['price'],
                    'buyerEmail': user.email,
                    'buyerName': user.displayName ?? '',
                    'deliveryAddress': addressCtrl.text.trim(),
                    'phone': phoneCtrl.text.trim(),
                    'preferredDate': selectedDate,
                    'timestamp': Timestamp.now(),
                    'status': 'Pending',
                  });

                  Navigator.pop(context);
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('✅ Order placed!')),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: primary,
                  minimumSize: const Size.fromHeight(50),
                ),
                child: const Text('Place Order'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
