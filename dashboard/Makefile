.PHONY: dev build build-local preview test clean deploy-local

# Vite dev server. Default base ("/") so localhost paths just work.
dev:
	VITE_BASE=/ npm run dev

# Production build (uses VITE_BASE=/CHARMPheno/ for GH Pages).
build:
	VITE_BASE=/CHARMPheno/ npm run build

# Local production build (serves at "/"). Use when you want to preview a built
# bundle without setting up a /CHARMPheno/ mount.
build-local:
	VITE_BASE=/ npm run build

# Vite preview server against the local build. Surfaces base-URL issues the
# dev server hides.
preview: build-local
	npm run preview -- --port 4173

test:
	npm test

clean:
	rm -rf dist node_modules

# Optional: manual push to gh-pages without going through CI. Requires gh-pages.
deploy-local: build
	npx gh-pages -d dist
