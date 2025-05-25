import 'package:flutter/material.dart';
import 'package:my_app/navigation.dart';


class ResultPage extends StatelessWidget {
  final int score;
  final int total;

  const ResultPage({super.key, required this.score, required this.total});

  @override
  Widget build(BuildContext context) {
    final percentage = (score / total * 100).round();

    return Scaffold(
      appBar: AppBar(
        title: const Text("Quiz Result"),
        backgroundColor: Colors.green[700],
        automaticallyImplyLeading: false, // 🔒 Hides the back button
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                "You scored $score out of $total",
                style:
                    const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 20),
              Text(
                "Your result: $percentage%",
                style: TextStyle(
                  fontSize: 20,
                  color: percentage >= 70 ? Colors.green : Colors.red,
                ),
              ),
              const SizedBox(height: 30),
              ElevatedButton.icon(
                icon: const Icon(Icons.replay),
                label: const Text("Try Again"),
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) =>
                        const NavigationPage(initialIndex: 2), // 2 = Quiz
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
