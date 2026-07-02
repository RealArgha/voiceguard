import React, { useEffect, useRef, useState } from 'react';
import { Animated, View, Text, StyleSheet, Dimensions } from 'react-native';
import Svg, { Circle } from 'react-native-svg';
import { Band, bandColor, colors } from '../constants/theme';

const { width: SCREEN_W } = Dimensions.get('window');

// Pixel 8: 412 dp wide — ring takes 60% of width, capped at 260
const SIZE   = Math.min(Math.round(SCREEN_W * 0.60), 260);
const RADIUS = Math.round(SIZE * 0.41);
const STROKE = Math.round(SIZE * 0.052);
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

  const color     = bandColor(band);
  const scoreFontSize = Math.round(SIZE * 0.22);
  const maxFontSize   = Math.round(SIZE * 0.058);
  const badgeFontSize = Math.round(SIZE * 0.048);

  return (
    <View style={{ width: SIZE, height: SIZE, alignItems: 'center', justifyContent: 'center' }}>
      <Svg width={SIZE} height={SIZE} style={StyleSheet.absoluteFill}>
        <Circle
          cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
          fill="none"
          stroke={colors.surface2}
          strokeWidth={STROKE}
        />
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

      <View style={{ alignItems: 'center', justifyContent: 'center' }} pointerEvents="none">
        <Text style={{ fontSize: scoreFontSize, fontWeight: '800', letterSpacing: -2, color, lineHeight: scoreFontSize * 1.1 }}>
          {displayScore}
        </Text>
        <Text style={{ fontSize: maxFontSize, color: colors.muted, marginTop: 2 }}>/100</Text>
        <View style={{ marginTop: 6, paddingHorizontal: 12, paddingVertical: 3,
                       borderRadius: 20, backgroundColor: color + '33' }}>
          <Text style={{ fontSize: badgeFontSize, fontWeight: '700', letterSpacing: 1.5,
                         textTransform: 'uppercase', color }}>
            {band}
          </Text>
        </View>
      </View>
    </View>
  );
}
