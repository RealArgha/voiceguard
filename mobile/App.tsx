/**
 * VoiceGuard mobile — customer-facing app.
 *
 * Flow:
 *  1. User taps "Call my bank" → opens native dialer with Twilio number.
 *  2. Twilio routes the call to our backend, which streams audio through
 *     the CNN + Whisper pipeline.
 *  3. This app subscribes to ws://<backend>/ws/dashboard and renders
 *     the live risk score for the user to see.
 *
 * Why this design (vs. embedding the Twilio Voice SDK natively):
 *  - Works inside Expo Go — no bare-workflow / Xcode / Android Studio
 *    setup needed for the hackathon demo.
 *  - User experience is "trustable third-party gauge" rather than
 *    "weird app that hijacked my dial pad" — actually a stronger pitch.
 *
 * To upgrade later: replace the Linking.openURL dialer with
 *   @twilio/voice-react-native-sdk for in-app VoIP. Same backend works.
 */

import { StatusBar } from 'expo-status-bar';
import CallScreen from './components/CallScreen';

export default function App() {
  return (
    <>
      <StatusBar style="light" />
      <CallScreen />
    </>
  );
}
