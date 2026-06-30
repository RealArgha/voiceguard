import React, { useEffect, useRef, useState } from 'react';
import { Animated, View, Text, StyleSheet } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import { Band, bandColor, colors } from '../constants/theme';

const SIZE   = 220;
const RADIUS = 90;
const STROKE = 12;
const CIRC   = 2 * Math.PI * RADIUS;

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

interface Props {
  score: number;
  band:  Band;
}

export default function RingGauge({ score, band }: Props) {
  const progress     = useRef(new Animated.Value(0)).current;
  const displayed    = useRef(new Animated.Value(0)).current;
  const [displayScore, setDisplayScore] = useState(0);

  useEffect(() => {
    Animated.timing(progress, {
      toValue:         score / 100,
      duration:        450,
      useNativeDriver: false,
    }).start();

    Animated.timing(displayed, {
      toValue:         score,
      duration:        450,
      useNativeDriver: false,
    }).start();
  }, [score]);

  useEffect(() => {
    const id = displayed.addListener(({ value }) => setDisplayScore(Math.round(value)));
    return () => displayed.removeListener(id);
  }, []);

  const strokeDashoffset = progress.interpolate({
    inputRange:  [0, 1],
    outputRange: [CIRC, 0],
  });

  const color = bandColor(band);

  return (
    <View style={styles.wrap}>
      <Svg width={SIZE} height={SIZE} style={styles.svg}>
        {/* background track */}
        <Circle
          cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
          fill="none"
          stroke={colors.surface2}
          strokeWidth={STROKE}
        />
        {/* progress arc — rotated so it starts at 12 o'clock */}
        <AnimatedCircle
          cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={CIRC}
          strokeDashoffset={strokeDashoffset}
          rotation="-90"
          origin={`${SIZE / 2}, ${SIZE / 2}`}
        />
      </Svg>

      {/* centre text */}
      <View style={styles.center} pointerEvents="none">
        <Text style={[styles.score, { color }]}>{displayScore}</Text>
        <Text style={styles.max}>/100</Text>
        <View style={[styles.badge, { backgroundColor: bandColor(band) + '33' }]}>
          <Text style={[styles.badgeText, { color }]}>{band}</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: SIZE, height: SIZE,
    alignItems: 'center', justifyContent: 'center',
  },
  svg: {
    position: 'absolute',
  },
  center: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  score: {
    fontSize: 52,
    fontWeight: '800',
    letterSpacing: -2,
    lineHeight: 56,
  },
  max: {
    fontSize: 13,
    color: colors.muted,
    marginTop: 2,
  },
  badge: {
    marginTop: 6,
    paddingHorizontal: 12,
    paddingVertical: 3,
    borderRadius: 20,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1.5,
    textTransform: 'uppercase',
  },
});
