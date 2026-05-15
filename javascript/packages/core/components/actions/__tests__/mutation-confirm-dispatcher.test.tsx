import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { MutationConfirmDispatcher } from '#core/components/actions/mutation-confirm-dispatcher';
import { buildWrapper } from '#core/test/wrappers/build-wrapper';
import { getBaseProviderWrapper } from '#core/test/wrappers/get-base-provider-wrapper';
import { getErrorProviderWrapper } from '#core/test/wrappers/get-error-provider-wrapper';
import { getRouterWrapper } from '#core/test/wrappers/get-router-wrapper';
import {
  createQueryMockRouter,
  getServiceProviderWrapper,
} from '#core/test/wrappers/get-service-provider-wrapper';

import type { ActionConfig } from '#core/components/actions/types';

function buildAction(): ActionConfig & {
  action: Extract<ActionConfig['action'], { type: 'mutation' }>;
  modal: Extract<ActionConfig['modal'], { type: 'confirm' }>;
} {
  return {
    display: { label: 'Kill' },
    action: { type: 'mutation', mutation: { mutationName: 'UpdateTriggerRun' } },
    modal: {
      type: 'confirm',
      header: { title: 'Confirm kill?' },
      button: { label: 'Kill it' },
      destructive: true,
    },
  };
}

describe('MutationConfirmDispatcher', () => {
  it('confirms → mutates with the record → closes', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const mockRequest = createQueryMockRouter({ UpdateTriggerRun: { triggerRun: {} } });

    render(
      <MutationConfirmDispatcher
        action={buildAction()}
        record={{ id: 'run-1' }}
        onClose={onClose}
      />,
      buildWrapper([
        getBaseProviderWrapper(),
        getErrorProviderWrapper(),
        getRouterWrapper({ location: '/test-ns' }),
        getServiceProviderWrapper({ request: mockRequest }),
      ])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Confirm kill?' });
    await user.click(within(dialog).getByRole('button', { name: 'Kill it' }));

    await waitFor(() => {
      expect(mockRequest).toHaveBeenCalledWith('UpdateTriggerRun', { id: 'run-1' });
    });
    await waitFor(() => {
      expect(onClose).toHaveBeenCalled();
    });
  });

  it('on mutation failure: dialog stays open, shows error banner, cancel re-enabled', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const mockRequest = createQueryMockRouter({
      UpdateTriggerRun: new Error('Service unavailable'),
    });

    render(
      <MutationConfirmDispatcher
        action={buildAction()}
        record={{ id: 'run-1' }}
        onClose={onClose}
      />,
      buildWrapper([
        getBaseProviderWrapper(),
        getErrorProviderWrapper(),
        getRouterWrapper({ location: '/test-ns' }),
        getServiceProviderWrapper({ request: mockRequest }),
      ])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Confirm kill?' });
    await user.click(within(dialog).getByRole('button', { name: 'Kill it' }));

    expect(await within(dialog).findByText(/Test error/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Cancel' })).toBeEnabled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('cancel triggers onClose without mutating', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const mockRequest = createQueryMockRouter({ UpdateTriggerRun: {} });

    render(
      <MutationConfirmDispatcher
        action={buildAction()}
        record={{ id: 'run-1' }}
        onClose={onClose}
      />,
      buildWrapper([
        getBaseProviderWrapper(),
        getErrorProviderWrapper(),
        getRouterWrapper({ location: '/test-ns' }),
        getServiceProviderWrapper({ request: mockRequest }),
      ])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Confirm kill?' });
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

    expect(onClose).toHaveBeenCalled();
    expect(mockRequest).not.toHaveBeenCalled();
  });
});
