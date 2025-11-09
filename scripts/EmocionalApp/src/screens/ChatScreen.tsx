/**
 * Pantalla de Chat Principal
 * Basada en el prototipo UX de Stitch
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  TextInput,
  FlatList,
  TouchableOpacity,
  Text,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { StackScreenProps } from '@react-navigation/stack';
import { Colors, Spacing, Typography, BorderRadius, Layout } from '../constants/DesignSystem';
import { RootStackParamList, Message } from '../types';
import ChatBubble from '../components/ChatBubble';
import { initLlama, LlamaContext } from 'llama.rn';

type Props = StackScreenProps<RootStackParamList, 'Chat'>;

const ChatScreen: React.FC<Props> = ({ route, navigation }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [charCount, setCharCount] = useState(0);

  const flatListRef = useRef<FlatList<Message>>(null);
  const llamaCtxRef = useRef<LlamaContext | null>(null);

  // Inicializar el modelo
  useEffect(() => {
    initializeModel();
    addWelcomeMessage();
  }, []);

  const initializeModel = async () => {
    try {
      const ctx = await initLlama({
        model: 'file://' + route.params.modelPath,
        n_ctx: 4096,
        n_threads: 6,
        n_gpu_layers: Platform.OS === 'ios' ? 20 : 0,
      });
      llamaCtxRef.current = ctx;
    } catch (error) {
      console.error('Error initializing model:', error);
    }
  };

  const addWelcomeMessage = () => {
    const welcomeMsg: Message = {
      id: 'welcome',
      text: "Hi there! I'm Aura, your personal AI companion. I'm here to listen and support you. Remember, our conversations are completely private and processed locally on your device. How are you feeling today?",
      isUser: false,
      timestamp: new Date(),
    };
    setMessages([welcomeMsg]);
  };

  const generateReply = async (prompt: string): Promise<string> => {
    const ctx = llamaCtxRef.current;
    if (!ctx) throw new Error('Modelo aún no cargado');

    const out = await ctx.completion({
      messages: [{ role: 'user', content: prompt }],
      n_predict: 400,
      temperature: 0.7,
    });

    return (
      (out as any).text ??
      (out as any).choices?.[0]?.message?.content ??
      ''
    );
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      text: input,
      isUser: true,
      timestamp: new Date(),
      status: 'sending',
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setCharCount(0);

    try {
      await new Promise(r => setTimeout(r, 300));

      setMessages(prev =>
        prev.map(m => (m.id === userMsg.id ? { ...m, status: 'sent' } : m)),
      );

      setIsBotTyping(true);
      const response = await generateReply(userMsg.text);

      const botMsg: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        isUser: false,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      console.error('LLM error:', err);
      setMessages(prev =>
        prev.map(m => (m.id === userMsg.id ? { ...m, status: 'error' } : m)),
      );
    } finally {
      setIsBotTyping(false);
    }
  };

  const handleInputChange = (text: string) => {
    if (text.length <= 500) {
      setInput(text);
      setCharCount(text.length);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.select({ ios: 'padding', android: undefined })}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 0}
    >
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.menuButton}
          onPress={() => navigation.navigate('Settings')}
          accessible
          accessibilityLabel="Abrir configuración"
          accessibilityRole="button"
        >
          <Text style={styles.menuIcon}>☰</Text>
        </TouchableOpacity>

        <View style={styles.headerCenter}>
          <Text style={styles.headerTitle}>Aura</Text>
          <View style={styles.statusContainer}>
            <View style={styles.statusDot} />
            <Text style={styles.statusText}>Online</Text>
          </View>
        </View>

        <TouchableOpacity
          style={styles.privacyButton}
          accessible
          accessibilityLabel="Privacidad activada"
          accessibilityRole="button"
        >
          <Text style={styles.privacyIcon}>🔒</Text>
        </TouchableOpacity>
      </View>

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={({ item }) => <ChatBubble message={item} />}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.messages}
        onContentSizeChange={() =>
          flatListRef.current?.scrollToEnd({ animated: true })
        }
        ListFooterComponent={
          isBotTyping ? (
            <View style={styles.typingIndicator}>
              <View style={styles.typingDots}>
                <View style={[styles.typingDot, styles.typingDot1]} />
                <View style={[styles.typingDot, styles.typingDot2]} />
                <View style={[styles.typingDot, styles.typingDot3]} />
              </View>
            </View>
          ) : null
        }
        removeClippedSubviews
        initialNumToRender={10}
        maxToRenderPerBatch={5}
        windowSize={7}
      />

      {/* Privacy Badge */}
      <View style={styles.privacyBadge}>
        <Text style={styles.privacyBadgeText}>
          Privado | Procesando localmente
        </Text>
      </View>

      {/* Input */}
      <View style={styles.inputContainer}>
        <TextInput
          value={input}
          onChangeText={handleInputChange}
          style={styles.input}
          placeholder="Type your message..."
          placeholderTextColor={Colors.textMuted}
          multiline
          maxLength={500}
        />
        
        <View style={styles.inputFooter}>
          <Text style={styles.charCount}>{charCount}/500</Text>
          
          <TouchableOpacity
            onPress={sendMessage}
            style={[
              styles.sendButton,
              !input.trim() && styles.sendButtonDisabled,
            ]}
            disabled={!input.trim()}
            accessible
            accessibilityLabel="Enviar mensaje"
            accessibilityRole="button"
          >
            <Text style={styles.sendButtonText}>▶</Text>
          </TouchableOpacity>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    height: Layout.headerHeight,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.md,
    backgroundColor: Colors.backgroundLight,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  menuButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  menuIcon: {
    fontSize: 24,
    color: Colors.text,
  },
  headerCenter: {
    flex: 1,
    alignItems: 'center',
  },
  headerTitle: {
    ...Typography.h4,
    color: Colors.text,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 2,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.online,
    marginRight: 6,
  },
  statusText: {
    ...Typography.caption,
    color: Colors.textSecondary,
  },
  privacyButton: {
    width: 40,
    height: 40,
    justifyContent: 'center',
    alignItems: 'center',
  },
  privacyIcon: {
    fontSize: 20,
  },
  messages: {
    paddingVertical: Spacing.md,
  },
  typingIndicator: {
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
  },
  typingDots: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.botBubble,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.sm,
    borderRadius: BorderRadius.lg,
    alignSelf: 'flex-start',
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.textSecondary,
    marginHorizontal: 2,
  },
  typingDot1: {
    opacity: 0.4,
  },
  typingDot2: {
    opacity: 0.7,
  },
  typingDot3: {
    opacity: 1,
  },
  privacyBadge: {
    backgroundColor: Colors.backgroundCard,
    paddingVertical: Spacing.xs,
    paddingHorizontal: Spacing.md,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  privacyBadgeText: {
    ...Typography.caption,
    color: Colors.accent,
  },
  inputContainer: {
    backgroundColor: Colors.backgroundLight,
    paddingHorizontal: Spacing.md,
    paddingVertical: Spacing.md,
    borderTopWidth: 1,
    borderTopColor: Colors.border,
  },
  input: {
    minHeight: 50,
    maxHeight: 120,
    backgroundColor: Colors.backgroundCard,
    borderRadius: BorderRadius.lg,
    paddingHorizontal: Spacing.md,
    paddingTop: Spacing.md,
    paddingBottom: Spacing.md,
    ...Typography.body,
    color: Colors.text,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  inputFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: Spacing.sm,
  },
  charCount: {
    ...Typography.caption,
    color: Colors.textMuted,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: Colors.primary,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: Colors.backgroundCard,
    opacity: 0.5,
  },
  sendButtonText: {
    color: Colors.text,
    fontSize: 20,
    fontWeight: 'bold',
  },
});

export default ChatScreen;

