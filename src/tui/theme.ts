import {
  type EditorTheme,
  type MarkdownTheme,
  type SelectListTheme,
  setCapabilityOverrides,
} from '@earendil-works/pi-tui';

// Enable terminal OSC 8 hyperlinks unless explicitly disabled or running in a dumb terminal.
// This allows links in markdown (e.g. model responses, web searches) to be clickable in terminal emulators.
if (process.env.TERM !== 'dumb' && process.env.PI_HYPERLINKS !== '0') {
  setCapabilityOverrides({ hyperlinks: true });
}

const colorEnabled = !process.env.NO_COLOR;

function sgr(open: number, close: number): (text: string) => string {
  return colorEnabled ? (text) => `\x1b[${open}m${text}\x1b[${close}m` : (text) => text;
}

export const style = {
  bold: sgr(1, 22),
  dim: sgr(2, 22),
  italic: sgr(3, 23),
  underline: sgr(4, 24),
  strikethrough: sgr(9, 29),
  red: sgr(31, 39),
  green: sgr(32, 39),
  yellow: sgr(33, 39),
  blue: sgr(34, 39),
  magenta: sgr(35, 39),
  cyan: sgr(36, 39),
  gray: sgr(90, 39),
};

export const markdownTheme: MarkdownTheme = {
  heading: (text) => style.bold(style.cyan(text)),
  link: style.cyan,
  linkUrl: style.gray,
  code: style.yellow,
  codeBlock: (text) => text,
  codeBlockBorder: style.gray,
  quote: style.italic,
  quoteBorder: style.gray,
  hr: style.gray,
  listBullet: style.cyan,
  bold: style.bold,
  italic: style.italic,
  strikethrough: style.strikethrough,
  underline: style.underline,
};

export const selectListTheme: SelectListTheme = {
  selectedPrefix: style.cyan,
  selectedText: style.cyan,
  description: style.gray,
  scrollInfo: style.gray,
  noMatch: style.gray,
};

export const editorTheme: EditorTheme = {
  borderColor: style.gray,
  selectList: selectListTheme,
};
