import { Tabs } from 'expo-router';
import { colors } from '../constants/theme';

export default function RootLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown:         false,
        tabBarStyle:         {
          backgroundColor:   colors.surface,
          borderTopColor:    colors.border,
          borderTopWidth:    1,
          height:            60,
          paddingBottom:     8,
        },
        tabBarActiveTintColor:   colors.accent,
        tabBarInactiveTintColor: colors.muted,
        tabBarLabelStyle:        { fontSize: 11, fontWeight: '600' },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{ title: 'Monitor', tabBarIcon: ({ color }) => <TabIcon label="🎙" color={color} /> }}
      />
      <Tabs.Screen
        name="history"
        options={{ title: 'History', tabBarIcon: ({ color }) => <TabIcon label="🕒" color={color} /> }}
      />
    </Tabs>
  );
}

function TabIcon({ label, color }: { label: string; color: string }) {
  const { Text } = require('react-native');
  return <Text style={{ fontSize: 18, opacity: color === colors.accent ? 1 : 0.5 }}>{label}</Text>;
}
