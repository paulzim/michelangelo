/**
 * Type-level tests asserting the {@link ActionConfig} union enforces what it
 * claims:
 *   - A custom modal cannot pair with an action.
 *   - A confirm modal requires an action to confirm.
 *   - Mutation and route actions can stand alone (no modal needed).
 *
 * Each `// @ts-expect-error` line is the assertion — if the type stops being
 * an error, TypeScript fails the build.
 *
 * This file contains type-level assertions only; it is excluded from vitest.
 */
/* eslint-disable local/no-module-scope-test-setup */

import type { ComponentType } from 'react';
import type {
  ActionComponentProps,
  ActionConfig,
  ActionHierarchy,
} from '#core/components/actions/types';

const display = { label: 'Action' };
const hierarchy: ActionHierarchy | undefined = undefined;

const mutationAction = {
  type: 'mutation' as const,
  mutation: { mutationName: 'X' },
};

const routeAction = {
  type: 'route' as const,
  route: '/somewhere',
};

const confirmModal = {
  type: 'confirm' as const,
  header: { title: 'Confirm?' },
  button: { label: 'OK' },
};

const customModal = {
  type: 'custom' as const,
  component: (() => null) as ComponentType<ActionComponentProps>,
};

// Valid: mutation + confirm modal.
const _validMutationWithConfirm: ActionConfig = {
  display,
  hierarchy,
  action: mutationAction,
  modal: confirmModal,
};

// Valid: route + confirm modal.
const _validRouteWithConfirm: ActionConfig = {
  display,
  action: routeAction,
  modal: confirmModal,
};

// Valid: mutation alone (no modal).
const _validMutationAlone: ActionConfig = {
  display,
  action: mutationAction,
};

// Valid: custom modal alone.
const _validCustomAlone: ActionConfig = {
  display,
  modal: customModal,
};

// Invalid: custom modal cannot pair with an action.
// @ts-expect-error — custom modal must stand alone, not pair with an action
const _invalidCustomWithAction: ActionConfig = {
  display,
  action: mutationAction,
  modal: customModal,
};

// Invalid: confirm modal alone is meaningless without an action to confirm.
// @ts-expect-error — confirm modal needs an action to confirm
const _invalidConfirmAlone: ActionConfig = {
  display,
  modal: confirmModal,
};

// Invalid: action of an unknown type.
const _invalidUnknownAction: ActionConfig = {
  display,
  action: {
    // @ts-expect-error — only 'mutation' or 'route' actions are allowed
    type: 'somethingElse',
  },
};

// Touch every binding so noUnusedLocals doesn't flag the assertions away.
void [
  _validMutationWithConfirm,
  _validRouteWithConfirm,
  _validMutationAlone,
  _validCustomAlone,
  _invalidCustomWithAction,
  _invalidConfirmAlone,
  _invalidUnknownAction,
];
