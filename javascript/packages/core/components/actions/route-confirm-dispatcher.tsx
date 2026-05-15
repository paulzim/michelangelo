import { useNavigate } from 'react-router-dom-v5-compat';

import { Banner } from '#core/components/banner/banner';
import { ConfirmDialog } from '#core/components/modal/confirm-dialog/confirm-dialog';

import type { ActionConfig, ConfirmModalConfig, Data, RouteActionConfig } from './types';

type Props<T extends Data> = {
  action: ActionConfig<T> & { action: RouteActionConfig; modal: ConfirmModalConfig };
  onClose: () => void;
};

/**
 * Renders a {@link ConfirmDialog} for an action that navigates to a route.
 * Confirming the dialog navigates; cancelling closes it.
 */
export function RouteConfirmDispatcher<T extends Data>({ action, onClose }: Props<T>) {
  const navigate = useNavigate();

  return (
    <ConfirmDialog
      isOpen
      onDismiss={onClose}
      onConfirm={() => navigate(action.action.route)}
      heading={action.modal.header.title}
      confirmLabel={action.modal.button.label}
      destructive={action.modal.destructive}
      size={action.modal.size}
    >
      {action.modal.banner && (
        <Banner kind={action.modal.banner.kind}>{action.modal.banner.content}</Banner>
      )}
      {action.modal.body}
    </ConfirmDialog>
  );
}
