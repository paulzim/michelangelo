import { useLocation } from 'react-router-dom-v5-compat';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { RouteConfirmDispatcher } from '#core/components/actions/route-confirm-dispatcher';
import { buildWrapper } from '#core/test/wrappers/build-wrapper';
import { getBaseProviderWrapper } from '#core/test/wrappers/get-base-provider-wrapper';
import { getRouterWrapper } from '#core/test/wrappers/get-router-wrapper';

import type { ActionConfig } from '#core/components/actions/types';

function buildAction(): ActionConfig & {
  action: Extract<ActionConfig['action'], { type: 'route' }>;
  modal: Extract<ActionConfig['modal'], { type: 'confirm' }>;
} {
  return {
    display: { label: 'Open detail' },
    action: { type: 'route', route: '/dest/page' },
    modal: {
      type: 'confirm',
      header: { title: 'Open detail page?' },
      button: { label: 'Open' },
    },
  };
}

// Renders the current location as text so the test can assert navigation.
function ShowLocation() {
  const { pathname } = useLocation();
  return <div data-testid="location">{pathname}</div>;
}

describe('RouteConfirmDispatcher', () => {
  it('confirms → navigates to the route', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <>
        <RouteConfirmDispatcher action={buildAction()} onClose={onClose} />
        <ShowLocation />
      </>,
      buildWrapper([getBaseProviderWrapper(), getRouterWrapper({ location: '/start' })])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Open detail page?' });
    await user.click(within(dialog).getByRole('button', { name: 'Open' }));

    expect(screen.getByTestId('location')).toHaveTextContent('/dest/page');
  });

  it('cancel triggers onClose without navigating', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <>
        <RouteConfirmDispatcher action={buildAction()} onClose={onClose} />
        <ShowLocation />
      </>,
      buildWrapper([getBaseProviderWrapper(), getRouterWrapper({ location: '/start' })])
    );

    const dialog = await screen.findByRole('dialog', { name: 'Open detail page?' });
    await user.click(within(dialog).getByRole('button', { name: 'Cancel' }));

    expect(onClose).toHaveBeenCalled();
    expect(screen.getByTestId('location')).toHaveTextContent('/start');
  });
});
