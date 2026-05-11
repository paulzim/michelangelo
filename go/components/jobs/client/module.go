package client

import (
	"github.com/michelangelo-ai/michelangelo/go/components/jobs/client/k8sengine"
	"github.com/michelangelo-ai/michelangelo/go/components/jobs/common/secrets"
	"github.com/michelangelo-ai/michelangelo/go/components/jobs/compute"
	"go.uber.org/fx"
)

// Module wires the federated jobs client with its default mapper impl.
//
// The default impl is k8sengine. It's bundled here so OSS apps get a working
// setup from one import. To swap impls (e.g. for tests, or an internal repo
// that wires its own LogPersistenceConfig), import the consumer-only pieces
// (NewClient, NewHelper, compute.Module, secrets.Module) à la carte and
// substitute a different *.Module providing types.Mapper.
var Module = fx.Options(
	fx.Provide(NewClient),
	fx.Provide(NewHelper),
	compute.Module,
	secrets.Module,
	k8sengine.Module,
)
