<script lang="ts">
  import SubTabs from '../SubTabs.svelte'
  import { subRoute } from '../router'
  import Atlas from './Atlas.svelte'
  import Compare from '../atlas/Compare.svelte'
</script>

<SubTabs top="atlas" />
{#if $subRoute === 'compare'}
  <Compare />
{:else}
  <Atlas />
{/if}
