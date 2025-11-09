/**
 * App Principal - Aplicación de Apoyo Emocional
 * Navegación actualizada con todas las pantallas
 */

import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { StatusBar } from 'react-native';
import { Colors } from './src/constants/DesignSystem';
import { RootStackParamList } from './src/types';
import { getUserSettings } from './src/utils/storage';

// Screens
import OnboardingScreen from './src/screens/OnboardingScreen';
import DownloadScreen from './src/screens/DownloadScreen';
import ChatScreen from './src/screens/ChatScreen';
import SettingsScreen from './src/screens/SettingsScreen';

const Stack = createStackNavigator<RootStackParamList>();

export default function App() {
  const [initialRoute, setInitialRoute] = useState<keyof RootStackParamList | null>(null);

  useEffect(() => {
    checkOnboardingStatus();
  }, []);

  const checkOnboardingStatus = async () => {
    try {
      const settings = await getUserSettings();
      
      if (!settings || !settings.hasCompletedOnboarding) {
        // Usuario nuevo, mostrar onboarding
        setInitialRoute('Onboarding');
      } else {
        // Usuario existente, ir directamente al chat o descarga
        setInitialRoute('Download');
      }
    } catch (error) {
      console.error('Error checking onboarding status:', error);
      setInitialRoute('Onboarding');
    }
  };

  if (!initialRoute) {
    // Mostrar splash screen o loading mientras se verifica
    return null;
  }

  return (
    <>
      <StatusBar
        barStyle="light-content"
        backgroundColor={Colors.background}
      />
      <NavigationContainer>
        <Stack.Navigator
          initialRouteName={initialRoute}
          screenOptions={{
            headerShown: false,
            cardStyle: { backgroundColor: Colors.background },
            animationEnabled: true,
          }}
        >
          <Stack.Screen name="Onboarding" component={OnboardingScreen} />
          <Stack.Screen name="Download" component={DownloadScreen} />
          <Stack.Screen name="Chat" component={ChatScreen} />
          <Stack.Screen name="Settings" component={SettingsScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}

