/**
 * RiskGauge — animated horizontal bar 0..100.
 * Plain RN, no external libs. Easy to swap for react-native-svg if you
 * want a circular gauge later.
 */

import { useEffect, useRef } from 'react';
import { Animated, Easing, StyleSheet, Text, View } from 'react-native';

type Props = { score: number; color: string };

export default function RiskGauge({ score, color }: Props) {
  const width = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(width, {
      toValue:  Math.max(0, Math.min(100, score)),
      duration: 400,
      easing:   Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
  }, [score]);

  return (
    <View>
      <Text style={[styles.score, { color }]}>
        {score.toFixed(0)}<Text style={styles.scoreOf}>/100</Text>
      </Text>
      <View style={styles.track}>
        <Animated.View style={[styles.fill, {
          width: width.interpolate({
            inputRange:  [0, 100],
            outputRange: ['0%', '100%'],
          }),
          backgroundColor: color,
        }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  score:   { fontSize: 48, fontWeight: '800', textAlign: 'center' },
  scoreOf: { fontSize: 18, color: '#64748b', fontWeight: '400' },
  track:   { height: 12, backgroundColor: '#0f172a',
             borderRadius: 6, overflow: 'hidden', marginTop: 8 },
  fill:    { height: '100%' },
});
