import { useMemo } from 'react';

import type { ActionConfig, Data, ResolvedActionItem } from './types';

/**
 * Resolves an array of {@link ActionConfig} into render-ready items. Pure —
 * does not navigate, mount components, or trigger mutations. The renderer's
 * `onSelect` callback decides what to do when an item is clicked.
 *
 * The returned items have stable identity across renders (memoized on
 * `actions`/`onSelect`) so consumers can compare items by reference — for
 * example, to track which item is hovered.
 */
export function useResolvedActionItems<T extends Data>(
  actions: ActionConfig<T>[],
  onSelect: (action: ActionConfig<T>) => void
): ResolvedActionItem[] {
  return useMemo(
    () =>
      actions.map((action) => {
        const matchingRule = action.disabled?.find((rule) => rule.condition);
        return {
          display: action.display,
          hierarchy: action.hierarchy,
          disabled: !!matchingRule,
          disabledMessage: matchingRule?.message,
          onClick: () => onSelect(action),
        };
      }),
    [actions, onSelect]
  );
}
