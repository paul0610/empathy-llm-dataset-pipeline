/**
 * Tipos TypeScript para la aplicación
 */

// Tipos de navegación
export type RootStackParamList = {
  Onboarding: undefined;
  Download: undefined;
  Chat: { modelPath: string };
  Settings: undefined;
};

// Tipos de mensajes
export type Message = {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
};

// Tipos de contacto de emergencia
export type EmergencyContact = {
  name: string;
  phone: string;
};

// Tipos de configuración de usuario
export type UserSettings = {
  emergencyContact?: EmergencyContact;
  hasCompletedOnboarding: boolean;
  acceptedTerms: boolean;
  conversationCount: number;
  lastConversationDate?: Date;
};

// Tipos de estado de descarga
export type DownloadState = {
  progress: number;
  speed: number;
  bytesWritten: number;
  totalBytes: number;
  status: 'idle' | 'downloading' | 'installing' | 'complete' | 'error';
  error?: string;
};

// Tipos de paso de onboarding
export type OnboardingStep = {
  id: number;
  title: string;
  description: string;
};

