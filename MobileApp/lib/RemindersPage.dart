import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter_datetime_picker_plus/flutter_datetime_picker_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:timezone/data/latest.dart' as tz;
import 'package:timezone/timezone.dart' as tz;

final FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin =
    FlutterLocalNotificationsPlugin();

class Reminder {
  int id;
  String title;
  String description;
  DateTime time;
  String repeat;

  Reminder({
    required this.id,
    required this.title,
    required this.description,
    required this.time,
    required this.repeat,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'title': title,
        'description': description,
        'time': time.toIso8601String(),
        'repeat': repeat,
      };

  static Reminder fromJson(Map<String, dynamic> json) => Reminder(
        id: json['id'],
        title: json['title'],
        description: json['description'],
        time: DateTime.parse(json['time']),
        repeat: json['repeat'],
      );
}

class RemindersPage extends StatefulWidget {
  @override
  _RemindersPageState createState() => _RemindersPageState();
}

class _RemindersPageState extends State<RemindersPage> {
  final List<Reminder> _reminders = [];
  int _nextId = 0;

  final List<String> titleSuggestions = [
    "Water Change",
    "Filter Cleaning",
    "Medication Dose",
    "pH Check",
    "Temperature Check",
    "Tank Inspection"
  ];
  final List<String> descSuggestions = [
    "Replace tank water",
    "Clean filter media",
    "Add treatment drops",
    "Measure pH level",
    "Verify 25–28°C",
    "Inspect for lesions"
  ];

  @override
  void initState() {
    super.initState();
    tz.initializeTimeZones();
    Future.microtask(_loadRemindersFromPrefs);
  }

  Future<void> _saveRemindersToPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final list = _reminders.map((r) => json.encode(r.toJson())).toList();
    await prefs.setStringList('reminders', list);
  }

  Future<void> _loadRemindersFromPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList('reminders') ?? [];
    _reminders.clear();
    for (var item in list) {
      final r = Reminder.fromJson(json.decode(item));
      _reminders.add(r);
      _scheduleNotification(r);
      _nextId = (_nextId < r.id + 1) ? r.id + 1 : _nextId;
    }
    setState(() {});
  }

  void _scheduleNotification(Reminder r) async {
    final android = AndroidNotificationDetails(
      'channel',
      'Reminders',
      importance: Importance.max,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );
    await flutterLocalNotificationsPlugin.zonedSchedule(
      r.id,
      r.title,
      r.description,
      tz.TZDateTime.from(r.time, tz.local),
      NotificationDetails(android: android),
      androidScheduleMode: AndroidScheduleMode.exactAllowWhileIdle,
      matchDateTimeComponents: r.repeat == 'Every Day'
          ? DateTimeComponents.time
          : r.repeat == 'Every Week'
              ? DateTimeComponents.dayOfWeekAndTime
              : r.repeat == 'Every Month'
                  ? DateTimeComponents.dayOfMonthAndTime
                  : null,
    );
  }

  void _showReminderDialog({Reminder? existing}) {
    final titleCtrl = TextEditingController(text: existing?.title);
    final descCtrl = TextEditingController(text: existing?.description);
    var selTime = existing?.time ?? DateTime.now().add(Duration(minutes: 1));
    var selRepeat = existing?.repeat ?? 'Only Once';

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.blue[50],
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      builder: (ctx) => Padding(
        padding: MediaQuery.of(ctx).viewInsets + EdgeInsets.all(16),
        child: StatefulBuilder(
          builder: (ctx, setSt) => Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                existing == null ? 'Add Reminder' : 'Edit Reminder',
                style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Colors.blueAccent),
              ),
              SizedBox(height: 12),
              _buildField(
                  ctrl: titleCtrl,
                  hint: 'Title',
                  chips: titleSuggestions,
                  onChip: (v) => setSt(() => titleCtrl.text = v)),
              SizedBox(height: 8),
              _buildField(
                  ctrl: descCtrl,
                  hint: 'Description',
                  chips: descSuggestions,
                  onChip: (v) => setSt(() => descCtrl.text = v)),
              SizedBox(height: 12),
              Row(children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      DatePicker.showDateTimePicker(ctx,
                          showTitleActions: true,
                          currentTime: selTime,
                          onConfirm: (d) => setSt(() => selTime = d));
                    },
                    icon: Icon(Icons.calendar_today, color: Colors.white),
                    label: Text('Pick Time',
                        style: TextStyle(color: Colors.white)),
                    style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blueAccent,
                        shape: StadiumBorder()),
                  ),
                ),
                SizedBox(width: 10),
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: selRepeat,
                    decoration: InputDecoration(
                        labelText: 'Repeat',
                        border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(12))),
                    onChanged: (v) => setSt(() => selRepeat = v!),
                    items: [
                      'Only Once',
                      'Every Day',
                      'Every Week',
                      'Every Month'
                    ]
                        .map((e) => DropdownMenuItem(value: e, child: Text(e)))
                        .toList(),
                  ),
                ),
              ]),
              SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: () {
                  if (titleCtrl.text.trim().isEmpty) {
                    ScaffoldMessenger.of(ctx).showSnackBar(
                        SnackBar(content: Text('Title cannot be empty')));
                    return;
                  }
                  final newR = Reminder(
                    id: existing?.id ?? _nextId++,
                    title: titleCtrl.text,
                    description: descCtrl.text,
                    time: selTime,
                    repeat: selRepeat,
                  );
                  if (existing == null)
                    _reminders.add(newR);
                  else {
                    final i = _reminders.indexWhere((x) => x.id == existing.id);
                    _reminders[i] = newR;
                  }
                  _scheduleNotification(newR);
                  _saveRemindersToPrefs();
                  Navigator.pop(ctx);
                  setState(() {});
                },
                icon: Icon(Icons.save, color: Colors.white),
                label: Text('Save', style: TextStyle(color: Colors.white)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blueAccent,
                ),
              ),
              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildField({
    required TextEditingController ctrl,
    required String hint,
    required List<String> chips,
    required void Function(String) onChip,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextField(
          controller: ctrl,
          decoration: InputDecoration(
            labelText: hint,
            fillColor: Colors.white,
            filled: true,
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            suffixIcon: IconButton(
                icon: Icon(Icons.clear), onPressed: () => ctrl.clear()),
          ),
        ),
        SizedBox(height: 8),
        Wrap(
          spacing: 8,
          children: chips
              .map((s) => ActionChip(
                    label: Text(s),
                    backgroundColor: Colors.blue[100],
                    onPressed: () => onChip(s),
                  ))
              .toList(),
        )
      ],
    );
  }

  void _deleteReminder(Reminder r) {
    flutterLocalNotificationsPlugin.cancel(r.id);
    setState(() => _reminders.removeWhere((x) => x.id == r.id));
    _saveRemindersToPrefs();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.blue[50],
      appBar: AppBar(
        title: Text('Reminders'),
        backgroundColor: Colors.blueAccent,
        elevation: 4,
        actions: [
          IconButton(
            icon: Icon(Icons.info_outline),
            onPressed: () => showAboutDialog(
              context: context,
              applicationName: 'Fish Care Reminders',
              applicationVersion: '1.0.0',
              children: [Text('Reminders for fish tank care.')],
            ),
          )
        ],
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: Colors.blueAccent,
        child: Icon(Icons.add),
        onPressed: () => _showReminderDialog(),
      ),
      body: _reminders.isEmpty
          ? Center(
              child: Text(
                "No reminders yet. Tap + to add.",
                style: TextStyle(fontSize: 16, color: Colors.grey[700]),
              ),
            )
          : ListView.builder(
              padding: EdgeInsets.symmetric(vertical: 8),
              itemCount: _reminders.length,
              itemBuilder: (_, i) {
                final r = _reminders[i];
                return Card(
                  margin: EdgeInsets.symmetric(vertical: 6, horizontal: 12),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12)),
                  elevation: 2,
                  child: ListTile(
                    leading: Icon(Icons.alarm, color: Colors.blueAccent),
                    title: Text(r.title,
                        style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Colors.blueAccent)),
                    subtitle: Text(
                      "${r.description}\n${r.time.toLocal().toString().split('.')[0]} (${r.repeat})",
                      style: TextStyle(color: Colors.grey[800]),
                    ),
                    isThreeLine: true,
                    trailing: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        IconButton(
                          icon: Icon(Icons.edit, color: Colors.blueGrey),
                          onPressed: () => _showReminderDialog(existing: r),
                        ),
                        IconButton(
                          icon: Icon(Icons.delete, color: Colors.red),
                          onPressed: () => _deleteReminder(r),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
    );
  }
}
