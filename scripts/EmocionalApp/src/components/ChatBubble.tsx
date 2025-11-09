/**
 * Componente de Burbuja de Chat Mejorado
 * Basado en el prototipo de chat principal
 */

import React from 'react';
import { View, Text, StyleSheet, Image } from 'react-native';
import Animated, { FadeInDown, FadeOutUp } from 'react-native-reanimated';
import { Colors, Typography, Spacing, BorderRadius, Shadows } from '../constants/DesignSystem';
import { Message } from '../types';

interface ChatBubbleProps {
  message: Message;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const { text, isUser, timestamp, status } = message;
  
  const bubbleStyle = isUser ? styles.user : styles.bot;
  const textColor = isUser ? Colors.text : Colors.text;

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('es-ES', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusIcon = () => {
    if (!status) return null;
    switch (status) {
      case 'sending':
        return '🕒';
      case 'error':
        return '❌';
      case 'sent':
        return '✓';
      default:
        return null;
    }
  };

  return (
    <Animated.View
      entering={FadeInDown.duration(300).springify()}
      exiting={FadeOutUp.duration(200)}
      style={styles.container}
      accessible
      accessibilityRole="text"
      accessibilityLabel={`Mensaje de ${isUser ? 'ti' : 'Aura'}: ${text}`}
    >
      {/* Avatar para mensajes del bot */}
      {!isUser && (
        <View style={styles.avatar}>
          <View style={styles.avatarGradient}>
            <Text style={styles.avatarText}>A</Text>
          </View>
        </View>
      )}

      <View style={[styles.bubbleContainer, bubbleStyle]}>
        <Text style={[styles.text, { color: textColor }]}>{text}</Text>
        
        <View style={styles.footer}>
          <Text style={styles.timestamp}>{formatTime(timestamp)}</Text>
          {status && (
            <Text style={styles.status}>{getStatusIcon()}</Text>
          )}
        </View>
      </View>

      {/* Avatar para mensajes del usuario */}
      {isUser && (
        <View style={styles.avatar}>
          <View style={styles.avatarUser}>
            <Text style={styles.avatarText}>U</Text>
          </View>
        </View>
      )}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    marginVertical: Spacing.sm,
    paddingHorizontal: Spacing.md,
  },
  avatar: {
    width: 36,
    height: 36,
    marginHorizontal: Spacing.sm,
  },
  avatarGradient: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.accent,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarUser: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    ...Typography.body,
    color: Colors.text,
    fontWeight: '600',
  },
  bubbleContainer: {
    maxWidth: '70%',
    padding: Spacing.md,
    borderRadius: BorderRadius.lg,
    ...Shadows.sm,
  },
  user: {
    backgroundColor: Colors.userBubble,
    alignSelf: 'flex-end',
    borderTopRightRadius: Spacing.xs,
  },
  bot: {
    backgroundColor: Colors.botBubble,
    alignSelf: 'flex-start',
    borderTopLeftRadius: Spacing.xs,
  },
  text: {
    ...Typography.body,
    color: Colors.text,
  },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: Spacing.xs,
    justifyContent: 'flex-end',
  },
  timestamp: {
    ...Typography.caption,
    color: Colors.textSecondary,
    marginRight: Spacing.xs,
  },
  status: {
    ...Typography.caption,
  },
});

export default ChatBubble;

