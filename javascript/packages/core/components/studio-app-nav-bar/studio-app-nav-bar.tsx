import { AppNavBar } from 'baseui/app-nav-bar';

import { Link } from '#core/components/link/link';

import type { NavItem } from 'baseui/app-nav-bar';

const DOCS_URL = 'https://michelangelo-ai.org/docs/';
const DOCS_NAV_ITEM = { label: 'Docs' };
const MAIN_NAV_ITEMS: NavItem[] = [DOCS_NAV_ITEM];

function mapMainNavItemToNode(item: NavItem) {
  if (item.label === DOCS_NAV_ITEM.label) {
    return (
      <Link
        href={DOCS_URL}
        overrides={{
          Link: {
            style: {
              color: 'inherit',
              ':hover': { color: 'inherit' },
              ':visited': { color: 'inherit' },
            },
          },
        }}
      >
        Docs
      </Link>
    );
  }

  return item.label;
}

export function StudioAppNavBar() {
  return (
    <AppNavBar
      mainItems={MAIN_NAV_ITEMS}
      mapItemToNode={mapMainNavItemToNode}
      title={
        <Link href="/" overrides={{ Link: { style: { ':hover': { textDecoration: 'unset' } } } }}>
          Michelangelo Studio
        </Link>
      }
    />
  );
}
