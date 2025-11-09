/**
 * Sistema de Diseño - Aplicación de Apoyo Emocional
 * Basado en los prototipos UX de Stitch
 */

export const Colors = {
  // Colores principales (oscuros y cálidos)
  background: '#1A2332',        // Azul oscuro principal
  backgroundLight: '#243447',   // Azul oscuro más claro
  backgroundCard: '#2A3B4F',    // Para cards y elementos elevados
  
  // Colores de acento
  primary: '#1DA1F2',           // Azul brillante (botones principales)
  primaryDark: '#0D8BD9',       // Azul más oscuro (hover)
  accent: '#4ECDC4',            // Turquesa (badges, highlights)
  accentGreen: '#2ECC71',       // Verde (success, online)
  
  // Colores de texto
  text: '#FFFFFF',              // Texto principal
  textSecondary: '#8B98A8',     // Texto secundario
  textMuted: '#5A6A7A',         // Texto deshabilitado
  
  // Colores de estado
  success: '#2ECC71',           // Verde
  warning: '#F39C12',           // Naranja
  error: '#E74C3C',             // Rojo
  info: '#3498DB',              // Azul info
  
  // Colores de chat
  userBubble: '#1DA1F2',        // Burbujas del usuario
  botBubble: '#3A4A5F',         // Burbujas del bot
  
  // Colores de UI
  border: '#3A4A5F',            // Bordes
  divider: '#2A3B4F',           // Divisores
  shadow: '#000000',            // Sombras
  overlay: 'rgba(0, 0, 0, 0.5)', // Overlay para modales
  
  // Colores especiales
  online: '#2ECC71',            // Indicador online
  offline: '#95A5A6',           // Indicador offline
  private: '#4ECDC4',           // Badge de privacidad
} as const;

export const Typography = {
  // Títulos
  h1: {
    fontSize: 32,
    lineHeight: 40,
    fontWeight: '700' as const,
    letterSpacing: -0.5,
  },
  h2: {
    fontSize: 24,
    lineHeight: 32,
    fontWeight: '700' as const,
    letterSpacing: -0.3,
  },
  h3: {
    fontSize: 20,
    lineHeight: 28,
    fontWeight: '600' as const,
    letterSpacing: -0.2,
  },
  h4: {
    fontSize: 18,
    lineHeight: 24,
    fontWeight: '600' as const,
  },
  
  // Cuerpo
  body: {
    fontSize: 16,
    lineHeight: 24,
    fontWeight: '400' as const,
  },
  bodyLarge: {
    fontSize: 18,
    lineHeight: 28,
    fontWeight: '400' as const,
  },
  bodySmall: {
    fontSize: 14,
    lineHeight: 20,
    fontWeight: '400' as const,
  },
  
  // Especiales
  caption: {
    fontSize: 12,
    lineHeight: 16,
    fontWeight: '400' as const,
  },
  button: {
    fontSize: 16,
    lineHeight: 24,
    fontWeight: '600' as const,
  },
  label: {
    fontSize: 14,
    lineHeight: 20,
    fontWeight: '500' as const,
  },
} as const;

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
} as const;

export const BorderRadius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 9999,
} as const;

export const Shadows = {
  sm: {
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  md: {
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
    elevation: 4,
  },
  lg: {
    shadowColor: Colors.shadow,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.2,
    shadowRadius: 16,
    elevation: 8,
  },
} as const;

export const Layout = {
  maxWidth: 600,              // Ancho máximo para contenido
  headerHeight: 60,           // Altura del header
  tabBarHeight: 60,           // Altura de la barra de navegación inferior
  inputHeight: 50,            // Altura de inputs
  buttonHeight: 50,           // Altura de botones
} as const;

export const Animation = {
  fast: 200,
  normal: 300,
  slow: 500,
} as const;

