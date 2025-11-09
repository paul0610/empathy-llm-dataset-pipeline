/**
 * Pantalla de Configuración/Ajustes
 * Basada en el prototipo UX de Stitch
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Modal,
  Alert,
} from 'react-native';
import { StackScreenProps } from '@react-navigation/stack';
import { Colors, Typography, Spacing, BorderRadius, Shadows } from '../constants/DesignSystem';
import { RootStackParamList, EmergencyContact } from '../types';
import Button from '../components/Button';
import Input from '../components/Input';
import {
  getEmergencyContact,
  saveEmergencyContact,
  deleteEmergencyContact,
  getConversationCount,
  resetConversationCount,
  clearAllData,
} from '../utils/storage';

type Props = StackScreenProps<RootStackParamList, 'Settings'>;

const SettingsScreen: React.FC<Props> = ({ navigation }) => {
  const [emergencyContact, setEmergencyContact] = useState<EmergencyContact | null>(null);
  const [conversationCount, setConversationCount] = useState(0);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [editName, setEditName] = useState('');
  const [editPhone, setEditPhone] = useState('');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    const contact = await getEmergencyContact();
    const count = await getConversationCount();
    setEmergencyContact(contact);
    setConversationCount(count);
  };

  const handleEditContact = () => {
    setEditName(emergencyContact?.name || '');
    setEditPhone(emergencyContact?.phone || '');
    setIsEditModalVisible(true);
  };

  const handleSaveContact = async () => {
    if (!editName.trim() || !editPhone.trim()) {
      Alert.alert('Error', 'Por favor completa todos los campos');
      return;
    }

    try {
      await saveEmergencyContact({
        name: editName.trim(),
        phone: editPhone.trim(),
      });
      await loadData();
      setIsEditModalVisible(false);
    } catch (error) {
      Alert.alert('Error', 'No se pudo guardar el contacto');
    }
  };

  const handleDeleteContact = () => {
    Alert.alert(
      'Eliminar Contacto',
      '¿Estás seguro de que deseas eliminar este contacto de emergencia?',
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Eliminar',
          style: 'destructive',
          onPress: async () => {
            try {
              await deleteEmergencyContact();
              await loadData();
            } catch (error) {
              Alert.alert('Error', 'No se pudo eliminar el contacto');
            }
          },
        },
      ],
    );
  };

  const handleDeleteConversation = () => {
    Alert.alert(
      'Eliminar Conversación',
      '¿Estás seguro? Esta acción no se puede deshacer.',
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Eliminar',
          style: 'destructive',
          onPress: async () => {
            // Aquí implementarías la lógica para eliminar una conversación específica
            Alert.alert('Info', 'Funcionalidad en desarrollo');
          },
        },
      ],
    );
  };

  const handleDeleteAll = () => {
    Alert.alert(
      'Borrar Todo',
      '¿Estás seguro de que deseas borrar todas las conversaciones? Esta acción no se puede deshacer.',
      [
        { text: 'Cancelar', style: 'cancel' },
        {
          text: 'Borrar Todo',
          style: 'destructive',
          onPress: async () => {
            try {
              await resetConversationCount();
              await loadData();
              Alert.alert('Éxito', 'Todas las conversaciones han sido eliminadas');
            } catch (error) {
              Alert.alert('Error', 'No se pudieron eliminar las conversaciones');
            }
          },
        },
      ],
    );
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}
          accessible
          accessibilityLabel="Volver"
          accessibilityRole="button"
        >
          <Text style={styles.backIcon}>←</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Configuración</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Emergency Contact Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Contacto de emergencia</Text>
          
          {emergencyContact ? (
            <View style={styles.contactCard}>
              <View style={styles.contactAvatar}>
                <Text style={styles.contactAvatarText}>👨‍⚕️</Text>
              </View>
              <View style={styles.contactInfo}>
                <Text style={styles.contactName}>{emergencyContact.name}</Text>
                <Text style={styles.contactPhone}>Tel: {emergencyContact.phone}</Text>
              </View>
            </View>
          ) : (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyText}>No hay contacto configurado</Text>
            </View>
          )}

          <View style={styles.buttonGroup}>
            <Button
              title="Editar"
              onPress={handleEditContact}
              variant="secondary"
              size="small"
              style={styles.buttonGroupItem}
            />
            {emergencyContact && (
              <Button
                title="Eliminar"
                onPress={handleDeleteContact}
                variant="danger"
                size="small"
                style={styles.buttonGroupItem}
              />
            )}
            {!emergencyContact && (
              <Button
                title="Agregar"
                onPress={handleEditContact}
                size="small"
                style={styles.buttonGroupItem}
              />
            )}
          </View>
        </View>

        {/* Conversation Management Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Gestión de conversaciones</Text>
          
          <View style={styles.statsCard}>
            <View style={styles.statRow}>
              <Text style={styles.statLabel}>Conversaciones totales</Text>
              <Text style={styles.statValue}>{conversationCount}</Text>
            </View>
            <View style={styles.statRow}>
              <Text style={styles.statLabel}>Última conversación</Text>
              <Text style={styles.statValue}>15 de mayo de 2024</Text>
            </View>
          </View>

          <Button
            title="Eliminar conversación específica"
            onPress={handleDeleteConversation}
            variant="secondary"
            fullWidth
            style={styles.sectionButton}
          />
          <Button
            title="Borrar todo"
            onPress={handleDeleteAll}
            variant="danger"
            fullWidth
          />
        </View>

        {/* Privacy & Security Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Privacidad y seguridad</Text>
          
          <View style={styles.privacyCard}>
            <View style={styles.privacyRow}>
              <Text style={styles.privacyLabel}>Cifrado</Text>
              <View style={styles.privacyStatus}>
                <Text style={styles.privacyStatusText}>Activado</Text>
                <Text style={styles.privacyStatusIcon}>✓</Text>
              </View>
            </View>
            
            <View style={styles.privacyRow}>
              <Text style={styles.privacyLabel}>Respaldos remotos</Text>
              <View style={[styles.privacyStatus, styles.privacyStatusInactive]}>
                <Text style={styles.privacyStatusText}>Desactivado</Text>
                <Text style={styles.privacyStatusIcon}>✓</Text>
              </View>
            </View>
            
            <View style={styles.privacyRow}>
              <Text style={styles.privacyLabel}>Almacenamiento</Text>
              <Text style={styles.privacyValue}>Local</Text>
            </View>
          </View>

          <Button
            title="Información detallada"
            onPress={() => Alert.alert('Info', 'Información detallada sobre privacidad')}
            variant="secondary"
            fullWidth
          />
        </View>

        {/* System Info Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Información del sistema</Text>
          
          <View style={styles.infoCard}>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Modelo de IA</Text>
              <Text style={styles.infoValue}>v2.1</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Espacio ocupado</Text>
              <Text style={styles.infoValue}>512 MB</Text>
            </View>
            
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Estado operativo</Text>
              <View style={styles.statusBadge}>
                <Text style={styles.statusBadgeText}>En línea</Text>
              </View>
            </View>
          </View>

          <Button
            title="Acerca de"
            onPress={() => Alert.alert('Acerca de', 'Emocional v1.0.0\nAplicación de apoyo emocional con IA')}
            variant="secondary"
            fullWidth
          />
        </View>
      </ScrollView>

      {/* Edit Contact Modal */}
      <Modal
        visible={isEditModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setIsEditModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>
              {emergencyContact ? 'Editar Contacto' : 'Agregar Contacto'}
            </Text>
            
            <Input
              label="Nombre"
              placeholder="Nombre completo"
              value={editName}
              onChangeText={setEditName}
              autoCapitalize="words"
            />
            
            <Input
              label="WhatsApp / Teléfono"
              placeholder="+1 (555) 123-4567"
              value={editPhone}
              onChangeText={setEditPhone}
              keyboardType="phone-pad"
            />

            <View style={styles.modalButtons}>
              <Button
                title="Cancelar"
                onPress={() => setIsEditModalVisible(false)}
                variant="ghost"
                style={styles.modalButton}
              />
              <Button
                title="Guardar"
                onPress={handleSaveContact}
                style={styles.modalButton}
              />
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    height: 60,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.md,
    backgroundColor: Colors.backgroundLight,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  backButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backIcon: {
    fontSize: 24,
    color: Colors.text,
  },
  headerTitle: {
    ...Typography.h3,
    color: Colors.text,
  },
  placeholder: {
    width: 40,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: Spacing.lg,
  },
  section: {
    marginBottom: Spacing.xl,
  },
  sectionTitle: {
    ...Typography.h3,
    color: Colors.text,
    marginBottom: Spacing.md,
  },
  contactCard: {
    flexDirection: 'row',
    backgroundColor: Colors.accent,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    marginBottom: Spacing.md,
    ...Shadows.md,
  },
  contactAvatar: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: Colors.backgroundLight,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.md,
  },
  contactAvatarText: {
    fontSize: 32,
  },
  contactInfo: {
    flex: 1,
    justifyContent: 'center',
  },
  contactName: {
    ...Typography.h4,
    color: Colors.text,
    marginBottom: Spacing.xs,
  },
  contactPhone: {
    ...Typography.body,
    color: Colors.text,
  },
  emptyCard: {
    backgroundColor: Colors.backgroundCard,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    alignItems: 'center',
    marginBottom: Spacing.md,
  },
  emptyText: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  buttonGroup: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  buttonGroupItem: {
    flex: 1,
    marginHorizontal: Spacing.xs,
  },
  statsCard: {
    backgroundColor: Colors.backgroundCard,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    marginBottom: Spacing.md,
  },
  statRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.sm,
  },
  statLabel: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  statValue: {
    ...Typography.body,
    color: Colors.text,
    fontWeight: '600',
  },
  sectionButton: {
    marginBottom: Spacing.md,
  },
  privacyCard: {
    backgroundColor: Colors.backgroundCard,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    marginBottom: Spacing.md,
  },
  privacyRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  privacyLabel: {
    ...Typography.body,
    color: Colors.text,
  },
  privacyStatus: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  privacyStatusInactive: {
    opacity: 0.7,
  },
  privacyStatusText: {
    ...Typography.bodySmall,
    color: Colors.success,
    marginRight: Spacing.xs,
  },
  privacyStatusIcon: {
    color: Colors.success,
  },
  privacyValue: {
    ...Typography.body,
    color: Colors.text,
    fontWeight: '600',
  },
  infoCard: {
    backgroundColor: Colors.backgroundCard,
    padding: Spacing.lg,
    borderRadius: BorderRadius.lg,
    marginBottom: Spacing.md,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: Spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  infoLabel: {
    ...Typography.body,
    color: Colors.textSecondary,
  },
  infoValue: {
    ...Typography.body,
    color: Colors.text,
    fontWeight: '600',
  },
  statusBadge: {
    backgroundColor: Colors.success,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.xs,
    borderRadius: BorderRadius.sm,
  },
  statusBadgeText: {
    ...Typography.bodySmall,
    color: Colors.text,
    fontWeight: '600',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: Colors.overlay,
    justifyContent: 'center',
    alignItems: 'center',
    padding: Spacing.lg,
  },
  modalContent: {
    width: '100%',
    maxWidth: 400,
    backgroundColor: Colors.backgroundLight,
    borderRadius: BorderRadius.lg,
    padding: Spacing.xl,
    ...Shadows.lg,
  },
  modalTitle: {
    ...Typography.h3,
    color: Colors.text,
    marginBottom: Spacing.lg,
    textAlign: 'center',
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: Spacing.md,
  },
  modalButton: {
    flex: 1,
    marginHorizontal: Spacing.xs,
  },
});

export default SettingsScreen;

