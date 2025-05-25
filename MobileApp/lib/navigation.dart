import 'package:flutter/material.dart';
import 'package:my_app/AddItemPage.dart';
import 'package:my_app/FishDiseaseDetectorPage.dart';
import 'package:my_app/ManageListingsPage.dart';
import 'package:my_app/RemindersPage.dart';
import 'package:my_app/fish_classifier_page.dart';
import 'package:my_app/settings_page.dart';

class NavigationPage extends StatefulWidget {
  final int initialIndex;
  const NavigationPage({super.key, this.initialIndex = 0});

  @override
  State<NavigationPage> createState() => _NavigationPageState();
}

class _NavigationPageState extends State<NavigationPage> {
  late int _currentIndex;

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
  }

  /// Builds a new page instance each time a tab is selected
  Widget _buildCurrentPage() {
    switch (_currentIndex) {
      case 0:
        return FishDiseaseDetectorPage();
      case 1:
        return FishClassifierPage(); // ✅ Added
      case 2:
        return RemindersPage();
      case 3:
        return ManageListingsPage();
      default:
        return SettingsPage();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _buildCurrentPage(),
      bottomNavigationBar: NavigationBar(
        height: 70,
        selectedIndex: _currentIndex,
        onDestinationSelected: (int index) {
          setState(() => _currentIndex = index);
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.bug_report),
            selectedIcon: Icon(Icons.bug_report_outlined),
            label: 'Disease',
          ),
          NavigationDestination(
            icon: Icon(Icons.pets), // ✅ Classifier icon
            selectedIcon: Icon(Icons.pets),
            label: 'Classifier',
          ),
          NavigationDestination(
            icon: Icon(Icons.alarm_outlined),
            selectedIcon: Icon(Icons.alarm),
            label: 'Reminders',
          ),
          NavigationDestination(
            icon: Icon(Icons.storefront),
            selectedIcon: Icon(Icons.storefront),
            label: 'Marketplace',
          ),
          NavigationDestination(
            icon: Icon(Icons.person),
            selectedIcon: Icon(Icons.person),
            label: 'profile',
          ),
        ],
      ),
    );
  }
}
