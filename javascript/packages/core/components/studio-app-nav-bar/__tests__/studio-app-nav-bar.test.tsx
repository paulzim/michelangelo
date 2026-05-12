import { render, screen } from '@testing-library/react';

import { buildWrapper } from '#core/test/wrappers/build-wrapper';
import { getBaseProviderWrapper } from '#core/test/wrappers/get-base-provider-wrapper';
import { getIconProviderWrapper } from '#core/test/wrappers/get-icon-provider-wrapper';
import { getRouterWrapper } from '#core/test/wrappers/get-router-wrapper';
import { StudioAppNavBar } from '../studio-app-nav-bar';

describe('StudioAppNavBar', () => {
  it('renders the Docs link in the app nav', () => {
    render(
      <StudioAppNavBar />,
      buildWrapper([getBaseProviderWrapper(), getIconProviderWrapper(), getRouterWrapper()])
    );

    const docsLink = screen.getByRole('link', { hidden: true, name: /docs/i });

    expect(docsLink).toHaveAttribute('href', 'https://michelangelo-ai.org/docs/');
    expect(docsLink).toHaveAttribute('target', '_blank');
    expect(docsLink).toHaveAttribute('rel', 'noopener noreferrer');
  });
});
