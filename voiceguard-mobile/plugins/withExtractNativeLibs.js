const { withAndroidManifest } = require('@expo/config-plugins');

// react-native-live-audio-stream ships .so files aligned to 4KB, not 16KB.
// Android 15 emulators and devices enforce 16KB page alignment by default.
// extractNativeLibs=true tells the installer to unpack .so files to disk
// instead of mapping them directly, bypassing the alignment requirement.
module.exports = function withExtractNativeLibs(config) {
  return withAndroidManifest(config, (config) => {
    config.modResults.manifest.application[0].$['android:extractNativeLibs'] = 'true';
    return config;
  });
};
