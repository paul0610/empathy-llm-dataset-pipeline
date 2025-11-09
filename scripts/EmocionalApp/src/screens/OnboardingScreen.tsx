/**
 * Pantalla de Onboarding/Bienvenida
 * Basada en el prototipo UX de Stitch
 */

import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { StackScreenProps } from '@react-navigation/stack';
import { Colors, Typography, Spacing, BorderRadius, Layout } from '../constants/DesignSystem';
import { RootStackParamList, EmergencyContact } from '../types';
import Button from '../components/Button';
import Input from '../components/Input';
import { saveUserSettings, saveEmergencyContact } from '../utils/storage';

type Props = StackScreenProps<RootStackParamList, 'Onboarding'>;

const { width } = Dimensions.get('window');

const OnboardingScreen: React.FC<Props> = ({ navigation }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [contactName, setContactName] = useState('');
  const [contactPhone, setContactPhone] = useState('');
  const scrollViewRef = useRef<ScrollView>(null);

  const handleNext = () => {
    if (currentStep < 2) {
      setCurrentStep(currentStep + 1);
      scrollViewRef.current?.scrollTo({
        x: width * (currentStep + 1),
        animated: true,
      });
    }
  };

  const handleSkip = () => {
    handleNext();
  };

  const handleFinish = async () => {
    try {
      // Guardar configuración de usuario
      await saveUserSettings({
        hasCompletedOnboarding: true,
        acceptedTerms,
        conversationCount: 0,
      });

      // Guardar contacto de emergencia si se proporcionó
      if (contactName.trim() && contactPhone.trim()) {
        await saveEmergencyContact({
          name: contactName.trim(),
          phone: contactPhone.trim(),
        });
      }

      // Navegar a la pantalla de descarga
      navigation.replace('Download');
    } catch (error) {
      console.error('Error saving onboarding data:', error);
    }
  };

  const renderDots = () => (
    <View style={styles.dotsContainer}>
      {[0, 1, 2].map((index) => (
        <View
          key={index}
          style={[
            styles.dot,
            currentStep === index && styles.dotActive,
          ]}
        />
      ))}
    </View>
  );

  const renderStep0 = () => (
    <View style={styles.stepContainer}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logo}>
            <Text style={styles.logoEmoji}>😊</Text>
          </View>
          <Text style={styles.appName}>Emocional</Text>
        </View>

        <Text style={styles.title}>¡Bienvenido!</Text>
        <Text style={styles.subtitle}>
          Tu bienestar emocional es nuestra prioridad. Descubre cómo Emocional puede ayudarte:
        </Text>

        {/* Features */}
        <View style={styles.featuresContainer}>
          <FeatureItem
            icon="✖️"
            title="IA Offline"
            description="Recibe apoyo personalizado en cualquier momento, sin necesidad de conexión a internet."
          />
          <FeatureItem
            icon="🔄"
            title="Privacidad Total"
            description="Tus datos permanecen en tu dispositivo, garantizando la máxima confidencialidad."
          />
          <FeatureItem
            icon="🔔"
            title="Detección Temprana"
            description="Detecta patrones de riesgo y recibe alertas tempranas para cuidar tu salud mental."
          />
        </View>

        {/* Privacy Badge */}
        <View style={styles.privacyBadge}>
          <Text style={styles.privacyText}>100% Privado - Todo local</Text>
        </View>
      </ScrollView>

      <View style={styles.footer}>
        {renderDots()}
        <Button
          title="Comenzar"
          onPress={handleNext}
          fullWidth
        />
      </View>
    </View>
  );

  const renderStep1 = () => (
    <View style={styles.stepContainer}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.title}>Términos y Condiciones</Text>
        <Text style={styles.subtitle}>
          Antes de continuar, por favor lee y acepta nuestros términos de uso:
        </Text>

        <View style={styles.termsContainer}>
          <Text style={styles.termsText}>
            • Emocional es una herramienta de apoyo emocional y no sustituye la terapia profesional.
            {'\n\n'}
            • En caso de emergencia o crisis, contacta inmediatamente a servicios de emergencia o un profesional de salud mental.
            {'\n\n'}
            • Tus conversaciones son completamente privadas y se procesan localmente en tu dispositivo.
            {'\n\n'}
            • No compartimos ni transmitimos tus datos personales a terceros.
          </Text>
        </View>

        <TouchableOpacity
          style={styles.checkboxContainer}
          onPress={() => setAcceptedTerms(!acceptedTerms)}
          accessible
          accessibilityRole="checkbox"
          accessibilityState={{ checked: acceptedTerms }}
        >
          <View style={[styles.checkbox, acceptedTerms && styles.checkboxChecked]}>
            {acceptedTerms && <Text style={styles.checkmark}>✓</Text>}
          </View>
          <Text style={styles.checkboxLabel}>
            Acepto los{' '}
            <Text style={styles.link}>términos y condiciones</Text>
            {' '}de uso
          </Text>
        </TouchableOpacity>

        <View style={styles.disclaimer}>
          <Text style={styles.disclaimerText}>
            Emocional no sustituye la terapia profesional
          </Text>
        </View>
      </ScrollView>

      <View style={styles.footer}>
        {renderDots()}
        <Button
          title="Continuar"
          onPress={handleNext}
          fullWidth
          disabled={!acceptedTerms}
        />
      </View>
    </View>
  );

  const renderStep2 = () => (
    <View style={styles.stepContainer}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.title}>Contacto de Emergencia (Opcional)</Text>
        <Text style={styles.subtitle}>
          Configura un contacto de confianza que pueda ayudarte en momentos difíciles.
        </Text>

        <Input
          label="Nombre"
          placeholder="Nombre completo"
          value={contactName}
          onChangeText={setContactName}
          autoCapitalize="words"
        />

        <Input
          label="WhatsApp / Teléfono"
          placeholder="+1 (555) 123-4567"
          value={contactPhone}
          onChangeText={setContactPhone}
          keyboardType="phone-pad"
        />

        <TouchableOpacity onPress={handleSkip} style={styles.skipButton}>
          <Text style={styles.skipText}>Omitir por ahora</Text>
        </TouchableOpacity>
      </ScrollView>

      <View style={styles.footer}>
        {renderDots()}
        <Button
          title="Finalizar"
          onPress={handleFinish}
          fullWidth
        />
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      <ScrollView
        ref={scrollViewRef}
        horizontal
        pagingEnabled
        scrollEnabled={false}
        showsHorizontalScrollIndicator={false}
        style={styles.scrollView}
      >
        {renderStep0()}
        {renderStep1()}
        {renderStep2()}
      </ScrollView>
    </View>
  );
};

// ============================================================================
// FEATURE ITEM COMPONENT
// ============================================================================

interface FeatureItemProps {
  icon: string;
  title: string;
  description: string;
}

const FeatureItem: React.FC<FeatureItemProps> = ({ icon, title, description }) => (
  <View style={styles.featureItem}>
    <View style={styles.featureIcon}>
      <Text style={styles.featureIconText}>{icon}</Text>
    </View>
    <View style={styles.featureContent}>
      <Text style={styles.featureTitle}>{title}</Text>
      <Text style={styles.featureDescription}>{description}</Text>
    </View>
  </View>
);

// ============================================================================
// STYLES
// ============================================================================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scrollView: {
    flex: 1,
  },
  stepContainer: {
    width,
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    padding: Spacing.lg,
  },
  header: {
    alignItems: 'center',
    marginBottom: Spacing.xl,
  },
  logo: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  logoEmoji: {
    fontSize: 32,
  },
  appName: {
    ...Typography.h3,
    color: Colors.text,
  },
  title: {
    ...Typography.h1,
    color: Colors.text,
    marginBottom: Spacing.md,
    textAlign: 'center',
  },
  subtitle: {
    ...Typography.body,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: Spacing.xl,
  },
  featuresContainer: {
    marginBottom: Spacing.xl,
  },
  featureItem: {
    flexDirection: 'row',
    marginBottom: Spacing.lg,
  },
  featureIcon: {
    width: 48,
    height: 48,
    borderRadius: BorderRadius.md,
    backgroundColor: Colors.backgroundCard,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  featureIconText: {
    fontSize: 24,
  },
  featureContent: {
    flex: 1,
  },
  featureTitle: {
    ...Typography.h4,
    color: Colors.text,
    marginBottom: Spacing.xs,
  },
  featureDescription: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
  },
  privacyBadge: {
    backgroundColor: Colors.backgroundCard,
    paddingVertical: Spacing.md,
    paddingHorizontal: Spacing.lg,
    borderRadius: BorderRadius.lg,
    alignSelf: 'center',
    borderWidth: 1,
    borderColor: Colors.accent,
  },
  privacyText: {
    ...Typography.body,
    color: Colors.accent,
    fontWeight: '600',
  },
  termsContainer: {
    backgroundColor: Colors.backgroundCard,
    padding: Spacing.lg,
    borderRadius: BorderRadius.md,
    marginBottom: Spacing.lg,
  },
  termsText: {
    ...Typography.bodySmall,
    color: Colors.textSecondary,
    lineHeight: 22,
  },
  checkboxContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: Spacing.lg,
  },
  checkbox: {
    width: 24,
    height: 24,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: Colors.border,
    marginRight: Spacing.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxChecked: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  checkmark: {
    color: Colors.text,
    fontSize: 16,
    fontWeight: 'bold',
  },
  checkboxLabel: {
    ...Typography.body,
    color: Colors.textSecondary,
    flex: 1,
  },
  link: {
    color: Colors.primary,
    textDecorationLine: 'underline',
  },
  disclaimer: {
    alignItems: 'center',
  },
  disclaimerText: {
    ...Typography.caption,
    color: Colors.textMuted,
    textAlign: 'center',
  },
  skipButton: {
    alignSelf: 'center',
    paddingVertical: Spacing.md,
  },
  skipText: {
    ...Typography.body,
    color: Colors.primary,
  },
  footer: {
    padding: Spacing.lg,
    paddingBottom: Spacing.xl,
  },
  dotsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: Spacing.lg,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.backgroundCard,
    marginHorizontal: 4,
  },
  dotActive: {
    backgroundColor: Colors.primary,
    width: 24,
  },
});

export default OnboardingScreen;

