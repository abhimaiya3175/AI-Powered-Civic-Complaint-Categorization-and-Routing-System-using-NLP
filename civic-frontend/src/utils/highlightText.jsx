const WORD_CHAR_REGEX = (() => {
  try {
    return new RegExp('[\\p{L}\\p{N}\\p{M}_]', 'u');
  } catch {
    return /[A-Za-z0-9_]/;
  }
})();

const LATIN_CHAR_REGEX = (() => {
  try {
    return new RegExp('\\p{Script=Latin}', 'u');
  } catch {
    return /[A-Za-z]/;
  }
})();

const COMBINING_MARK_REGEX = (() => {
  try {
    return new RegExp('\\p{M}', 'u');
  } catch {
    return /[\u0300-\u036f]/;
  }
})();

const COMBINING_MARK_GLOBAL_REGEX = (() => {
  try {
    return new RegExp('\\p{M}', 'gu');
  } catch {
    return /[\u0300-\u036f]/g;
  }
})();

const GRAPHEME_SEGMENTER = typeof Intl !== 'undefined' && Intl.Segmenter
  ? new Intl.Segmenter(undefined, { granularity: 'grapheme' })
  : null;

const REGEX_CACHE_LIMIT = 50;
const regexCache = new Map();

const escapeRegExp = (value) => String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
const escapeCharClass = (value) => String(value).replace(/[\\\]-]/g, '\\$&');

const normalizeNfc = (value) => String(value ?? '').normalize('NFC');

const getCachedRegex = (pattern, flags) => {
  const cacheKey = `${flags}:${pattern}`;
  const cached = regexCache.get(cacheKey);
  if (cached) {
    cached.lastIndex = 0;
    regexCache.delete(cacheKey);
    regexCache.set(cacheKey, cached);
    return cached;
  }

  const regex = new RegExp(pattern, flags);
  regexCache.set(cacheKey, regex);

  if (regexCache.size > REGEX_CACHE_LIMIT) {
    const oldestKey = regexCache.keys().next().value;
    regexCache.delete(oldestKey);
  }

  return regex;
};

const getGraphemeSegments = (text) => {
  if (!text) return [];

  if (GRAPHEME_SEGMENTER) {
    return Array.from(GRAPHEME_SEGMENTER.segment(text), ({ segment, index }) => ({ segment, index }));
  }

  const segments = [];
  let segment = '';
  let segmentStart = 0;

  for (let index = 0; index < text.length;) {
    const codePoint = text.codePointAt(index);
    const char = String.fromCodePoint(codePoint);
    const charLength = char.length;

    if (!segment) {
      segment = char;
      segmentStart = index;
    } else if (COMBINING_MARK_REGEX.test(char)) {
      segment += char;
    } else {
      segments.push({ segment, index: segmentStart });
      segment = char;
      segmentStart = index;
    }

    index += charLength;
  }

  if (segment) {
    segments.push({ segment, index: segmentStart });
  }

  return segments;
};

const buildNormalizedTextIndex = (source) => {
  const normalizedParts = [];
  const indexMap = [];

  getGraphemeSegments(source).forEach(({ segment, index }) => {
    const normalizedSegment = normalizeNfc(segment);
    const originalEnd = index + segment.length;

    normalizedParts.push(normalizedSegment);

    for (let offset = 0; offset < normalizedSegment.length; offset += 1) {
      indexMap.push({ start: index, end: originalEnd });
    }
  });

  return {
    text: normalizedParts.join(''),
    indexMap,
  };
};

const getPreviousChar = (text, index) => {
  if (index <= 0) return '';

  const current = text.charCodeAt(index - 1);
  const previous = index > 1 ? text.charCodeAt(index - 2) : 0;
  const isLowSurrogate = current >= 0xdc00 && current <= 0xdfff;
  const hasHighSurrogateBefore = previous >= 0xd800 && previous <= 0xdbff;

  return isLowSurrogate && hasHighSurrogateBefore
    ? text.slice(index - 2, index)
    : text.slice(index - 1, index);
};

const getNextChar = (text, index) => {
  if (index >= text.length) return '';

  const current = text.charCodeAt(index);
  const next = index + 1 < text.length ? text.charCodeAt(index + 1) : 0;
  const isHighSurrogate = current >= 0xd800 && current <= 0xdbff;
  const hasLowSurrogateAfter = next >= 0xdc00 && next <= 0xdfff;

  return isHighSurrogate && hasLowSurrogateAfter
    ? text.slice(index, index + 2)
    : text.slice(index, index + 1);
};

const isWordChar = (char) => Boolean(char && WORD_CHAR_REGEX.test(char));

const hasNonLatinWordSignal = (value) => Array.from(value)
  .some((char) => isWordChar(char) && !LATIN_CHAR_REGEX.test(char) && !/[0-9_]/.test(char));

const foldLatinCase = (value) => Array.from(value)
  .map((char) => (LATIN_CHAR_REGEX.test(char) ? char.toLocaleLowerCase('en-US') : char))
  .join('');

const foldModelFeature = (value) => foldLatinCase(String(value ?? '').normalize('NFD'))
  .replace(COMBINING_MARK_GLOBAL_REGEX, '')
  .normalize('NFC');

const buildLatinCasePattern = (char) => {
  const variants = [...new Set([
    char,
    char.toLocaleLowerCase('en-US').normalize('NFC'),
    char.toLocaleUpperCase('en-US').normalize('NFC'),
  ])].filter(Boolean);

  if (variants.length === 1) {
    return escapeRegExp(char);
  }

  if (variants.every((variant) => Array.from(variant).length === 1)) {
    return `[${variants.map(escapeCharClass).join('')}]`;
  }

  return `(?:${variants.map(escapeRegExp).join('|')})`;
};

const buildKeywordPattern = (keyword) => {
  let previousWasWhitespace = false;

  return Array.from(keyword).map((char) => {
    if (/\s/u.test(char)) {
      if (previousWasWhitespace) return '';
      previousWasWhitespace = true;
      return '\\s+';
    }

    previousWasWhitespace = false;
    if (LATIN_CHAR_REGEX.test(char)) return buildLatinCasePattern(char);
    return escapeRegExp(char);
  })
  .join('');
};

const normalizeKeywords = (keywords) => {
  if (!Array.isArray(keywords)) return [];

  const seen = new Set();
  return keywords
    .map((keyword) => normalizeNfc(keyword).trim())
    .filter(Boolean)
    .map((keyword) => ({
      keyword,
      pattern: buildKeywordPattern(keyword),
      key: foldLatinCase(keyword).replace(/\s+/g, ' '),
      featureKey: foldModelFeature(keyword).replace(/\s+/g, ' '),
      allowFeatureWordFallback: hasNonLatinWordSignal(keyword) && !/\s/u.test(keyword),
    }))
    .filter(({ key }) => {
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((a, b) => b.keyword.length - a.keyword.length);
};

const hasWordBoundaries = (text, start, end) => {
  const matchedStart = getNextChar(text, start);
  const matchedEnd = getPreviousChar(text, end);
  const previous = getPreviousChar(text, start);
  const next = getNextChar(text, end);

  const leftBoundaryOk = !isWordChar(matchedStart) || !isWordChar(previous);
  const rightBoundaryOk = !isWordChar(matchedEnd) || !isWordChar(next);

  return leftBoundaryOk && rightBoundaryOk;
};

const overlaps = (a, b) => a.start < b.end && b.start < a.end;

const getWordRanges = (text) => {
  const ranges = [];
  let rangeStart = null;

  for (let index = 0; index < text.length;) {
    const char = getNextChar(text, index);
    const charLength = char.length || 1;

    if (isWordChar(char)) {
      if (rangeStart === null) rangeStart = index;
    } else if (rangeStart !== null) {
      ranges.push({ start: rangeStart, end: index });
      rangeStart = null;
    }

    index += charLength;
  }

  if (rangeStart !== null) {
    ranges.push({ start: rangeStart, end: text.length });
  }

  return ranges;
};

const getOriginalRange = (indexMap, normalizedStart, normalizedEnd) => {
  const first = indexMap[normalizedStart];
  const last = indexMap[normalizedEnd - 1];

  if (!first || !last) return null;

  return {
    start: first.start,
    end: last.end,
  };
};

export function getHighlightRanges(text, keywords) {
  const source = text == null ? '' : String(text);
  if (!source) return [];

  const normalizedKeywords = normalizeKeywords(keywords);
  if (normalizedKeywords.length === 0) return [];

  const normalizedSource = buildNormalizedTextIndex(source);
  if (!normalizedSource.text) return [];

  const combinedPattern = normalizedKeywords
    .map(({ pattern }) => pattern)
    .filter(Boolean)
    .join('|');
  if (!combinedPattern) return [];

  let regex;
  try {
    regex = getCachedRegex(`(?=(${combinedPattern}))`, 'gu');
  } catch {
    return [];
  }

  const candidates = [];
  let match;

  while ((match = regex.exec(normalizedSource.text)) !== null) {
    const matchedText = match[1];
    const normalizedStart = match.index;
    const normalizedEnd = normalizedStart + (matchedText?.length || 0);

    if (matchedText && hasWordBoundaries(normalizedSource.text, normalizedStart, normalizedEnd)) {
      const originalRange = getOriginalRange(normalizedSource.indexMap, normalizedStart, normalizedEnd);
      if (originalRange) {
        candidates.push({
          ...originalRange,
          length: normalizedEnd - normalizedStart,
        });
      }
    }

    const nextChar = getNextChar(normalizedSource.text, normalizedStart);
    regex.lastIndex = normalizedStart + (nextChar.length || 1);
  }

  const fallbackKeywords = normalizedKeywords.filter(({ allowFeatureWordFallback, featureKey }) => (
    allowFeatureWordFallback && featureKey.length >= 2
  ));

  if (fallbackKeywords.length > 0) {
    getWordRanges(normalizedSource.text).forEach((wordRange) => {
      const word = normalizedSource.text.slice(wordRange.start, wordRange.end);
      const wordFeatureKey = foldModelFeature(word);

      const matchedKeyword = fallbackKeywords.find(({ featureKey }) => (
        wordFeatureKey === featureKey || wordFeatureKey.startsWith(featureKey)
      ));

      if (!matchedKeyword) return;

      const originalRange = getOriginalRange(normalizedSource.indexMap, wordRange.start, wordRange.end);
      if (originalRange) {
        candidates.push({
          ...originalRange,
          length: wordRange.end - wordRange.start,
        });
      }
    });
  }

  const selected = [];
  candidates
    .sort((a, b) => b.length - a.length || a.start - b.start)
    .forEach((candidate) => {
      if (!selected.some((range) => overlaps(range, candidate))) {
        selected.push(candidate);
      }
    });

  return selected.sort((a, b) => a.start - b.start);
}

export function highlightText(text, keywords) {
  const source = text == null ? '' : String(text);
  const ranges = getHighlightRanges(source, keywords);

  if (ranges.length === 0) {
    return <>{source}</>;
  }

  const nodes = [];
  let cursor = 0;

  ranges.forEach((range, index) => {
    if (range.start > cursor) {
      nodes.push(source.slice(cursor, range.start));
    }

    nodes.push(
      <mark key={`highlight-${range.start}-${range.end}-${index}`} className="text-highlight reason-highlight">
        {source.slice(range.start, range.end)}
      </mark>
    );

    cursor = range.end;
  });

  if (cursor < source.length) {
    nodes.push(source.slice(cursor));
  }

  return <>{nodes}</>;
}
