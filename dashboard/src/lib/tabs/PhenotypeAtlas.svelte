<script lang="ts">
  import SubTabs from '../SubTabs.svelte'
  import { subRoute, TOP_TABS } from '../router'
  import Atlas from './Atlas.svelte'
  import Compare from '../atlas/Compare.svelte'

  const title = TOP_TABS.find((t) => t.id === 'atlas')!.label
</script>

<header class="top-head">
  <h1>{title}</h1>
  <SubTabs top="atlas" />
</header>

{#if $subRoute === 'compare'}
  <Compare />
{:else}
  <Atlas />
{/if}

<style>
  .top-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    padding-bottom: 1.25rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--rule);
  }
  .top-head h1 {
    margin: 0;
  }
</style>
