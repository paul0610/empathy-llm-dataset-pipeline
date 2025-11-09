/**
 * Utilidades para almacenamiento local (AsyncStorage)
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { UserSettings, EmergencyContact } from '../types';

const KEYS = {
  USER_SETTINGS: '@emotional_app:user_settings',
  EMERGENCY_CONTACT: '@emotional_app:emergency_contact',
  CONVERSATIONS: '@emotional_app:conversations',
  MODEL_PATH: '@emotional_app:model_path',
};

// ============================================================================
// USER SETTINGS
// ============================================================================

export const getUserSettings = async (): Promise<UserSettings | null> => {
  try {
    const data = await AsyncStorage.getItem(KEYS.USER_SETTINGS);
    return data ? JSON.parse(data) : null;
  } catch (error) {
    console.error('Error getting user settings:', error);
    return null;
  }
};

export const saveUserSettings = async (settings: UserSettings): Promise<void> => {
  try {
    await AsyncStorage.setItem(KEYS.USER_SETTINGS, JSON.stringify(settings));
  } catch (error) {
    console.error('Error saving user settings:', error);
    throw error;
  }
};

export const updateUserSettings = async (
  updates: Partial<UserSettings>,
): Promise<void> => {
  try {
    const current = await getUserSettings();
    const updated = { ...current, ...updates };
    await saveUserSettings(updated as UserSettings);
  } catch (error) {
    console.error('Error updating user settings:', error);
    throw error;
  }
};

// ============================================================================
// EMERGENCY CONTACT
// ============================================================================

export const getEmergencyContact = async (): Promise<EmergencyContact | null> => {
  try {
    const data = await AsyncStorage.getItem(KEYS.EMERGENCY_CONTACT);
    return data ? JSON.parse(data) : null;
  } catch (error) {
    console.error('Error getting emergency contact:', error);
    return null;
  }
};

export const saveEmergencyContact = async (
  contact: EmergencyContact,
): Promise<void> => {
  try {
    await AsyncStorage.setItem(KEYS.EMERGENCY_CONTACT, JSON.stringify(contact));
  } catch (error) {
    console.error('Error saving emergency contact:', error);
    throw error;
  }
};

export const deleteEmergencyContact = async (): Promise<void> => {
  try {
    await AsyncStorage.removeItem(KEYS.EMERGENCY_CONTACT);
  } catch (error) {
    console.error('Error deleting emergency contact:', error);
    throw error;
  }
};

// ============================================================================
// CONVERSATIONS
// ============================================================================

export const getConversationCount = async (): Promise<number> => {
  try {
    const data = await AsyncStorage.getItem(KEYS.CONVERSATIONS);
    return data ? parseInt(data, 10) : 0;
  } catch (error) {
    console.error('Error getting conversation count:', error);
    return 0;
  }
};

export const incrementConversationCount = async (): Promise<void> => {
  try {
    const count = await getConversationCount();
    await AsyncStorage.setItem(KEYS.CONVERSATIONS, (count + 1).toString());
  } catch (error) {
    console.error('Error incrementing conversation count:', error);
    throw error;
  }
};

export const resetConversationCount = async (): Promise<void> => {
  try {
    await AsyncStorage.setItem(KEYS.CONVERSATIONS, '0');
  } catch (error) {
    console.error('Error resetting conversation count:', error);
    throw error;
  }
};

// ============================================================================
// MODEL PATH
// ============================================================================

export const getModelPath = async (): Promise<string | null> => {
  try {
    return await AsyncStorage.getItem(KEYS.MODEL_PATH);
  } catch (error) {
    console.error('Error getting model path:', error);
    return null;
  }
};

export const saveModelPath = async (path: string): Promise<void> => {
  try {
    await AsyncStorage.setItem(KEYS.MODEL_PATH, path);
  } catch (error) {
    console.error('Error saving model path:', error);
    throw error;
  }
};

// ============================================================================
// CLEAR ALL DATA
// ============================================================================

export const clearAllData = async (): Promise<void> => {
  try {
    await AsyncStorage.multiRemove([
      KEYS.USER_SETTINGS,
      KEYS.EMERGENCY_CONTACT,
      KEYS.CONVERSATIONS,
      // NO eliminamos MODEL_PATH para no tener que re-descargar el modelo
    ]);
  } catch (error) {
    console.error('Error clearing all data:', error);
    throw error;
  }
};

